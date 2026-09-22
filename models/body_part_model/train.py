import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import models, transforms

DATASET_PATH = Path("datasets/dataset_for_first_model.csv")
DATA_ROOT = Path("datasets")
IMG_SIZE = 288               
BATCH = 16
EPOCHS = 30                  
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(DATASET_PATH)

def row_label(r):
    if r["body_spine"] == 1: return 0
    if r["body_hip_right"] == 1: return 1
    if r["body_hip_left"] == 1: return 2
    return -1

df["class_id"] = df.apply(row_label, axis=1)
df = df[df["class_id"] >= 0].reset_index(drop=True)

CLASS_NAMES = ["spine", "hip_right", "hip_left"]
print(f"Всего: {len(df)}")
print(df["class_id"].value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES))))

# StratifiedGroupKFold держит баланс классов И не пускает один study в разные фолды
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(sgkf.split(df, df["class_id"], groups=df["study_id"]))
trainval_idx, test_idx = splits[0]
trainval = df.iloc[trainval_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# из trainval ещё раз отрезаем val тем же способом
sgkf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, val_idx = next(sgkf2.split(trainval, trainval["class_id"], groups=trainval["study_id"]))
train_df = trainval.iloc[tr_idx].reset_index(drop=True)
val_df   = trainval.iloc[val_idx].reset_index(drop=True)

def print_dist(name, d):
    c = d["class_id"].value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES)))
    print(f"{name}: n={len(d)} | {dict(c)}")

print_dist("train", train_df)
print_dist("val  ", val_df)
print_dist("test ", test_df)

assert not (set(train_df.study_id) & set(val_df.study_id))
assert not (set(train_df.study_id) & set(test_df.study_id))
assert not (set(val_df.study_id)   & set(test_df.study_id))

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

        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()

        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        img = self._pad_to_square(img)

        if self.train:
            img = transforms.RandomAffine(
                degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)
            )(img)

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0

        # RandomErasing только на train (после приведения к [0,1])
        if self.train and np.random.rand() < 0.25:
            x = transforms.RandomErasing(p=1.0, scale=(0.02, 0.08), value=0.0)(x)

        y = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return x, y


train_ds = DXADataset(train_df, train=True)
val_ds   = DXADataset(val_df,   train=False)
test_ds  = DXADataset(test_df,  train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)


model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

old_conv = model.features[0][0]
new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=False)

with torch.no_grad():
    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
model.features[0][0] = new_conv
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model = model.to(DEVICE)

class_counts = train_df["class_id"].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * len(class_weights)   # нормировка
class_weights = class_weights.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights,label_smoothing=LABEL_SMOOTHING)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
patience = 6
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step()
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch:2d}  lr={lr_now:.2e}  "
          f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
          f"val loss={va_loss:.4f} acc={va_acc:.3f}")
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), "best_model.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"early stopping на эпохе {epoch}")
            break

# Тест 
model.load_state_dict(torch.load("best_model.pt"))

def predict_with_tta(loader):
    """TTA: усредняем логиты оригинала и горизонтально отражённого.
    Для spine/hflip безвредно, для hip_right/hip_left — сознательно ломает сторону,
    поэтому здесь НЕ используем hflip. Ограничимся одним прямым проходом.
    Если хочешь TTA — используй мультикроп (см. ниже).
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, 1).cpu().numpy()
            y_pred.extend(probs.argmax(1).tolist())
            y_prob.extend(probs.tolist())
            y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

y_true, y_pred, y_prob = predict_with_tta(test_loader)
test_acc = (y_true == y_pred).mean()
print(f"\nTEST acc={test_acc:.3f}")

print("\nConfusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(cm)
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))