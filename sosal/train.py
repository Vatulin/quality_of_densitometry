import os
import time
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
import timm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from main import load_and_merge_data

# ==========================================
# 1. Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==========================================
# 2. Чтение DICOM
# ==========================================
def read_dicom_image(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
        
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    
    if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
        img = np.max(img) - img
        
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    else:
        img = np.zeros_like(img)
        
    img = img.astype(np.uint8)
    img = np.stack([img] * 3, axis=-1)
    return img

# ==========================================
# 3. Dataset
# ==========================================
class SpineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['full_path']
        label = float(row['is_incorrect'])
        
        image = read_dicom_image(img_path)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 4. Аугментации (ОПТИМИЗИРОВАНО: 256x256)
# ==========================================
train_transform = A.Compose([
    A.Resize(256, 256), # Уменьшили с 512 до 256 для скорости
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==========================================
# 5. Поиск оптимального порога
# ==========================================
def find_best_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return best_threshold

# ==========================================
# 6. Цикл обучения (ОПТИМИЗИРОВАНО)
# ==========================================
def train_fold(fold, train_df, val_df, device, epochs=20, batch_size=8):
    train_dataset = SpineDataset(train_df, transform=train_transform)
    val_dataset = SpineDataset(val_df, transform=val_transform)
    
    class_counts = train_df['is_incorrect'].value_counts()
    class_weights = 1.0 / class_counts
    sample_weights = train_df['is_incorrect'].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # ОПТИМИЗИРОВАНО: Используем легкую ResNet18 вместо тяжелой EfficientNet-B3
    model = timm.create_model('resnet18', pretrained=True, num_classes=1).to(device)
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Валидация
        model.eval()
        val_preds_probs, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images).squeeze(-1)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds_probs.extend(probs)
                val_targets.extend(labels.numpy())
                
        val_preds_probs = np.array(val_preds_probs)
        val_targets = np.array(val_targets)
        
        current_best_thresh = find_best_threshold(val_targets, val_preds_probs)
        binary_preds = (val_preds_probs >= current_best_thresh).astype(int)
        current_f1 = f1_score(val_targets, binary_preds, zero_division=0)
        
        # Вывод прогресса каждую эпоху
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {current_f1:.4f} | Time: {elapsed:.1f}s")
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = current_best_thresh
            torch.save(model.state_dict(), f"spine_model_fold_{fold}.pth")
            
    print(f"✅ Fold {fold} завершен. Best Val F1: {best_f1:.4f} (при пороге {best_threshold:.2f})\n")
    return best_f1, best_threshold

# ==========================================
# 7. Обучение финальной модели
# ==========================================
def train_final_model(df, device, epochs=20, batch_size=8):
    print(f"\n{'='*40}")
    print(f"Обучение финальной модели на ВСЕХ {len(df)} данных...")
    print(f"{'='*40}")
    
    final_dataset = SpineDataset(df, transform=train_transform)
    final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = timm.create_model('resnet18', pretrained=True, num_classes=1).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for images, labels in final_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        elapsed = time.time() - start_time
        print(f"  Final Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(final_loader):.4f} | Time: {elapsed:.1f}s")
        
    torch.save(model.state_dict(), "final_spine_model.pth")
    print("✅ Финальная модель сохранена как final_spine_model.pth")

# ==========================================
# 8. Главная функция
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    if device.type == 'cpu':
        print("⚠️  Обучение на CPU будет идти медленнее, чем на GPU. Ждите вывода в консоль.")
    
    df = load_and_merge_data()
    print(f"Всего записей: {len(df)}, Нарушений (1): {df['is_incorrect'].sum()}")
    
    # 1. Кросс-валидация
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['is_incorrect'])):
        print(f"\n🚀 Старт обучения Fold {fold}...")
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        score, thresh = train_fold(fold, train_df, val_df, device=device, epochs=20)
        fold_scores.append(score)
        
    print(f"\n{'='*40}")
    print(f"🏆 Средний F1-Score по кросс-валидации: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
    print(f"{'='*40}")
    
    # 2. Финальное обучение
    train_final_model(df, device=device, epochs=20)

if __name__ == '__main__':
    main()