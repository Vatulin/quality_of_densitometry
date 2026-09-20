import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# BASE_DIR = r"C:\Users\Sony\Downloads"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
EXCEL_PATH = os.path.join(SCRIPT_DIR, 'dataset_with_artefacts.xlsx')
OUTPUT_EXCEL = os.path.join(SCRIPT_DIR, 'artifact_predictions_result.xlsx')

BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
IMG_SIZE = 384 

def read_dxa_dicom(path):
    try:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        
        p_min, p_max = np.percentile(img, (1, 99))
        img = np.clip(img, p_min, p_max)
        img = (img - p_min) / (p_max - p_min + 1e-8)
        
        img_uint8 = (img * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_uint8)
        
        return np.stack([img_clahe] * 3, axis=-1)
    except Exception as e:
        print(f"Ошибка чтения файла {path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

class DXAArtifactDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row['rel_path']).replace('\\', '/')
        label = int(row['artefacts'])
        
        full_path = os.path.join(BASE_DIR, rel_path)
        img = read_dxa_dicom(full_path)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, torch.tensor(label, dtype=torch.float32), rel_path

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images).squeeze(-1)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    model.eval()
    all_labels, all_probs, all_paths = [], [], []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            if np.ndim(probs) == 0:
                probs = np.array([probs])
                
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            
    return np.array(all_labels), np.array(all_probs), all_paths

def find_best_threshold(y_true, y_probs):
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.35, 0.65, 0.02):
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH, sheet_name='Лист1')
    df['artefacts'] = df['artefacts'].fillna(0).astype(int)
    
    if 'body_spine' in df.columns:
        df_target = df[df['body_spine'] == 1].copy().reset_index(drop=True)
    else:
        df_target = df.copy()
        
    print(f"Всего снимков для анализа: {len(df_target)}")
    print(f"Наличие артефактов: {df_target['artefacts'].sum()} ({df_target['artefacts'].mean()*100:.1f}%)")
    
    train_df, val_df = train_test_split(
        df_target, test_size=0.2, random_state=42, stratify=df_target['artefacts']
    )
    
    train_loader = DataLoader(DXAArtifactDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(DXAArtifactDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Используемое устройство: {device}")
    
    model = timm.create_model('resnet18', pretrained=True, num_classes=1)
    model.to(device)
    
    num_pos = train_df['artefacts'].sum()
    num_neg = len(train_df) - num_pos
    raw_weight = num_neg / (num_pos + 1e-5)
    clipped_weight = min(raw_weight, 3.0) 
    pos_weight = torch.tensor([clipped_weight], dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    best_auc = 0.0
    best_model_thresh = 0.5
    
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_labels, val_probs, _ = validate(model, val_loader, device)
        
        auc = roc_auc_score(val_labels, val_probs)
        thresh, f1 = find_best_threshold(val_labels, val_probs)
        acc = accuracy_score(val_labels, (val_probs >= thresh).astype(int))
        
        scheduler.step()
        print(f"Эпоха {epoch+1:02d}/{EPOCHS} | Loss: {train_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} (Thresh: {thresh:.2f}) | Acc: {acc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_thresh = thresh
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_threshold': float(best_model_thresh),
            }, 'best_artifact_model.pth')
            print("  💾 Модель успешно сохранена!")

    print(f"\nОбучение завершено. Лучший AUC: {best_auc:.4f}")

    checkpoint = torch.load('best_artifact_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt_thresh = checkpoint.get('best_threshold', 0.5)
    
    val_labels, val_probs, val_paths = validate(model, val_loader, device)
    binary_preds = (val_probs >= opt_thresh).astype(int)

    report_df = pd.DataFrame({
        'path_to_study': val_paths,
        'quality_class': binary_preds, 
        'artifact_probability': np.round(val_probs, 4),
        'processing_status': 'Success'
    })

    report_df.to_excel(OUTPUT_EXCEL, index=False)
