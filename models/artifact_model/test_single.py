import os
import cv2
import numpy as np
import torch
import pydicom
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings('ignore')
IMG_SIZE = 384
MODEL_PATH = 'best_artifact_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Строго те же трансформации, что и при обучении
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def read_image(path):
    """Универсальная читалка: поддерживает и DICOM, и обычные фото"""
    if path.lower().endswith(('.dcm', '.dicom')):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        p_min, p_max = np.percentile(img, (1, 99))
        img = np.clip(img, p_min, p_max)
        img = (img - p_min) / (p_max - p_min + 1e-8)
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось прочитать изображение. Проверьте путь или формат.")
        img_uint8 = img

    # Применяем тот же фильтр контраста (CLAHE), что и при обучении
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_uint8)
    return np.stack([img_clahe] * 3, axis=-1)

def load_trained_model():
    model = timm.create_model('resnet18', pretrained=False, num_classes=1)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Файл {MODEL_PATH} не найден! Запустите скрипт из папки с моделью.")
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    thresh = checkpoint.get('best_threshold', 0.5)
    print(f"✅ Модель готова! Рабочий порог отсечения: {thresh:.2f}\n")
    return model, thresh

def predict_single_image(model, img_path, threshold):
    try:
        img = read_image(img_path)
        augmented = val_transform(image=img)
        tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logit = model(tensor).squeeze(-1)
            prob = torch.sigmoid(logit).item()
            
        has_artifact = prob >= threshold
        
        if has_artifact:
            print(f"РЕЗУЛЬТАТ: ОБНАРУЖЕН АРТЕФАКТ")
        else:
            print(f"ЕЗУЛЬТАТ: Снимок чистый (Норма)")
            
        print(f"Вероятность наличия артефакта: {prob*100:.1f}%")
        
    except Exception as e:
        print(f"шибка при обработке файла: {e}")
if __name__ == "__main__":
    try:
        model, opt_thresh = load_trained_model()
        
        print("Вставляйте пути к изображениям (.dcm, .jpg, .png) для проверки.")
        print("Для выхода напишите 'exit'.\n")
        
        while True:
            path = input("Путь к фото: ").strip(' "\'')
            
            if path.lower() in ['exit', 'выход', 'quit', 'q']:
                print("Завершение работы.")
                break
                
            if not os.path.exists(path):
                print("Файл не найден! Убедитесь, что путь скопирован верно.\n")
                continue
                
            predict_single_image(model, path, opt_thresh)
            
    except Exception as e:
        print(f"Критическая ошибка: {e}")