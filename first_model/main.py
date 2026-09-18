import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models, transforms

DATASET_PATH = Path("datasets/dataset_for_first_model.csv")
DATA_ROOT = Path("datasets")
IMG_SIZE = 224
BATCH = 16
EPOCHS = 20
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(DATASET_PATH)

def row_label(r):
    """0 = spine, 1 = hip (любой), -1 = нет метки"""
    if r["body_spine"] == 1:
        return 0
    if r["body_hip_right"] == 1 or r["body_hip_left"] == 1:
        return 1
    return -1

df["class_id"] = df.apply(row_label, axis=1)
df = df[df["class_id"] >= 0].reset_index(drop=True)

CLASS_NAMES = ["spine", "hip"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(df, groups=df["study_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# из train отрезаем val
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, val_idx = next(gss2.split(train_df, groups=train_df["study_id"]))
val_df = train_df.iloc[val_idx].reset_index(drop=True)
train_df = train_df.iloc[tr_idx].reset_index(drop=True)

print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
assert not (set(train_df.study_id) & set(val_df.study_id))
assert not (set(train_df.study_id) & set(test_df.study_id))
assert not (set(val_df.study_id) & set(test_df.study_id))

class DXADataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _pad_to_square(img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ds = pydicom.dcmread(str(DATA_ROOT / row["rel_path"]))
        arr = ds.pixel_array.astype(np.float32)

        # min-max нормализация
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()

        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        img = self._pad_to_square(img)

        if self.train:
            img = transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))(img)

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0
        y = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return x, y


train_ds = DXADataset(train_df, train=True)
val_ds = DXADataset(val_df, train=False)
test_ds = DXADataset(test_df, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model = model.to(DEVICE)

# веса классов на случай дисбаланса
class_counts = train_df["class_id"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Обучение
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


best_val = 0.0
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    scheduler.step()
    print(f"epoch {epoch:2d}  train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
          f"val loss={va_loss:.4f} acc={va_acc:.3f}")
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), "best_model.pt")

# Тест
model.load_state_dict(torch.load("best_model.pt"))
test_loss, test_acc = run_epoch(test_loader, train=False)
print(f"\nTEST loss={test_loss:.4f} acc={test_acc:.3f}")

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        pred = model(x).argmax(1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.tolist())

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))