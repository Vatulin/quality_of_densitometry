import pandas as pd
import pydicom
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import os

# Константы
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = 'C:\MIREA\Hacaton\MosHacaton\dataset\dataset.csv'
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_COLS = ['spine', 'right_hip', 'left_hip']

class DicomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw_path = str(row['rel_path']).strip().replace('\\', '/')
        full_path = os.path.normpath(os.path.join(BASE_DIR, raw_path))
        
        # 1. Извлекаем индекс класса из One-Hot колонок (0: spine, 1: right_hip, 2: left_hip)
        label = np.argmax(row[LABEL_COLS].values.astype(np.float32))
        
        # 2. Чтение DICOM файла
        dicom = pydicom.dcmread(full_path)
        image = dicom.pixel_array
        
        # 3. Нормализация пикселей в диапазон 0-255 (с учетом возможных 12/16-битных снимков)
        image = image - np.min(image)
        if np.max(image) != 0:
            image = image / np.max(image)
        image = (image * 255).astype(np.uint8)
        
        # 4. Перевод в 3 канала для преобученных сетей
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 5. Аугментация
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)

def train_model():
    df = pd.read_csv(CSV_PATH, encoding='cp1251')

    df = df.dropna(subset=['rel_path'])

# Принудительно приводим колоноку к строковому типу
    df['rel_path'] = df['rel_path'].astype(str)
    
    # Создаем вспомогательную колонку для стратификации при сплите
    df['target'] = df[LABEL_COLS].values.argmax(axis=1)
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['target'], random_state=42
    )
    
    # Преобразования изображений
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    train_loader = DataLoader(DicomDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DicomDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False)
    
    # Предобученная модель EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Заморозка весов для Transfer Learning
    for param in model.parameters():
        param.requires_grad = False
        
    # Замена классификатора под 3 класса
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(LABEL_COLS))
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    print(f"Запуск обучения на {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        acc = 100 * val_correct / val_total
        print(f"Эпоха [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {acc:.2f}%")
        
    torch.save(model.state_dict(), 'weights/anatomical_region_model.pth')
    print("Веса сохранены в 'anatomical_region_model.pth'")

if __name__ == '__main__':
    train_model()