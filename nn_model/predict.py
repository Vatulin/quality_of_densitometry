import os
import sys
import argparse
import pydicom
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Словарь соответствия индексов класса и их названий
CLASSES = {
    0: 'spine (Позвоночник)',
    1: 'right_hip (Правое бедро)',
    2: 'left_hip (Левое бедро)'
}

# Путь к сохраненным весам модели по умолчанию
MODEL_WEIGHTS_PATH = 'weights/anatomical_region_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms():
    """Трансформации для валидации/инференса (совпадают с обучением)"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def load_model(weights_path, num_classes=3):
    """Загрузка архитектуры EfficientNet-B0 и подгрузка весов"""
    model = models.efficientnet_b0(weights=None)
    
    # Заменяем финальный классификатор под наше количество классов
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл с весами модели '{weights_path}' не найден!")
        
    # Загружаем веса
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_dicom(dicom_path):
    """Чтение DICOM и подготовка тензора для нейросети"""
    if not os.path.exists(dicom_path):
        raise FileNotFoundError(f"DICOM файл по пути '{dicom_path}' не найден!")

    # 1. Чтение DICOM
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)

    # 2. Мин-макс нормализация пикселей в диапазон [0, 255]
    img_min, img_max = np.min(image), np.max(image)
    if img_max != img_min:
        image = (image - img_min) / (img_max - img_min) * 255.0
    else:
        image = np.zeros_like(image)
    image = image.astype(np.uint8)

    # 3. Перевод из 1-канального оттенка серого в 3-канальный RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # 4. Аугментации / Ресайз / Нормализация PyTorch
    transforms = get_transforms()
    augmented = transforms(image=image)
    image_tensor = augmented['image'].unsqueeze(0)  # Добавляем batch dimension: [1, 3, 224, 224]

    return image_tensor.to(DEVICE)


def predict(dicom_path, model_path=MODEL_WEIGHTS_PATH):
    """Основная функция классификации анатомического отдела"""
    print(f"Загрузка модели из {model_path}...")
    model = load_model(model_path)

    print(f"Обработка DICOM файла: {dicom_path}...")
    input_tensor = preprocess_dicom(dicom_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class_id = torch.argmax(probabilities).item()

    predicted_label = CLASSES[predicted_class_id]
    confidence = probabilities[predicted_class_id].item() * 100

    print("\n" + "=" * 50)
    print(f"ОПРЕДЕЛЕННЫЙ ОТДЕЛ: {predicted_label}")
    print(f"Уверенность модели: {confidence:.2f}%")
    print("=" * 50)
    print("\nРаспределение вероятностей по всем классам:")
    for idx, label in CLASSES.items():
        prob = probabilities[idx].item() * 100
        print(f" - {label:<28}: {prob:6.2f}%")
    
    return predicted_class_id, predicted_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Классификация анатомического отдела DICOM исследования.")
    parser.add_argument('--path', type=str, required=True, help="Путь к DICOM файлу (.dcm)")
    parser.add_argument('--model', type=str, default=MODEL_WEIGHTS_PATH, help="Путь к .pth файлу весов модели")

    args = parser.parse_args()
    predict(args.path, args.model)