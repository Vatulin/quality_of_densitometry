"""
Обучение модели контроля качества укладки проксимального отдела бедра.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
)
from torchvision import models, transforms

warnings.filterwarnings("ignore")

# ================== CONFIG ==================
CSV_PATH      = Path("datasets/dataset_for_hip_model.csv")
DATA_ROOT     = Path("datasets")
CKPT_PATH     = Path("best_hip_quality.pt")
IMG_SIZE      = 384
BATCH         = 8
EPOCHS        = 50
LR            = 2e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
PATIENCE      = 12
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEED          = 42
NUM_WORKERS   = 0   # на Windows проще 0; если хотите 2 — оставьте и используйте main-guard
LABEL_COLS    = ["positioning_rotation", "roi_correctness"]


# ================== DATASET ==================
class HipQualityDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_dicom(path):
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return (arr * 255).astype(np.uint8)

    @staticmethod
    def _pad_to_square(img):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas

    def __getitem__(self, i):
        row = self.df.iloc[i]

        arr = self._load_dicom(DATA_ROOT / row["rel_path"])
        img = Image.fromarray(arr).convert("L")
        img = self._pad_to_square(img)

        if self.train:
            img = transforms.RandomAffine(
                degrees=7,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )(img)

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0

        if self.train:
            if random.random() < 0.5:
                x = torch.clamp(x * random.uniform(0.85, 1.15), 0.0, 1.0)
            if random.random() < 0.5:
                mean = x.mean()
                x = torch.clamp(
                    (x - mean) * random.uniform(0.85, 1.15) + mean, 0.0, 1.0
                )
            if random.random() < 0.25:
                x = transforms.RandomErasing(
                    p=1.0, scale=(0.02, 0.06), value=0.0
                )(x)

        x = (x - 0.5) / 0.25
        side = int(row["side_id"])

        y = torch.zeros(len(LABEL_COLS), dtype=torch.float32)
        m = torch.zeros(len(LABEL_COLS), dtype=torch.float32)
        for j, c in enumerate(LABEL_COLS):
            v = row[c]
            if pd.notna(v):
                y[j] = float(v)
                m[j] = 1.0

        return x, torch.tensor(side, dtype=torch.long), y, m


# ================== MODEL ==================
class HipQualityModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", n_side=2, side_dim=32):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.side_emb = nn.Embedding(n_side, side_dim)
        self.dropout  = nn.Dropout(0.5)
        self.head     = nn.Linear(feat_dim + side_dim, len(LABEL_COLS))

    def forward(self, x, side):
        feats  = self.backbone(x)
        side_f = self.side_emb(side)
        h = torch.cat([feats, side_f], dim=1)
        h = self.dropout(h)
        return self.head(h)


# ================== UTILS ==================
def masked_bce_loss(logits, targets, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def compute_pos_weight(d, device):
    weights = []
    for c in LABEL_COLS:
        s = d[c].dropna()
        pos = (s == 1).sum()
        neg = (s == 0).sum()
        w = neg / max(pos, 1)
        weights.append(min(w, 20.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ys, ms = [], [], []
    for x, side, y, m in loader:
        x, side = x.to(device), side.to(device)
        with autocast(enabled=(device == "cuda")):
            logits = model(x, side)
        p = torch.sigmoid(logits).float().cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
        ms.append(m.numpy())
    return (np.concatenate(probs),
            np.concatenate(ys),
            np.concatenate(ms))


def best_threshold(y_true, y_prob, grid=np.linspace(0.05, 0.95, 91)):
    if y_true.sum() == 0:
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def bootstrap_ci(y, p, metric, n=1000, alpha=0.05, seed=42):
    if len(y) < 4:
        return (float("nan"), float("nan"))
    scores = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        try:
            scores.append(metric(y[idx], p[idx]))
        except Exception:
            pass
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 100 * alpha / 2)),
            float(np.percentile(scores, 100 * (1 - alpha / 2))))


# ================== MAIN ==================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"Device: {DEVICE}")

    # ---------- 1. Данные ----------
    df = pd.read_csv(CSV_PATH)
    df["study_id"] = df["study_id"].astype(str).str.strip()
    df["side_id"] = (df["side"] == "right").astype(int)

    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        uniq = set(df[c].dropna().unique().tolist())
        assert uniq.issubset({0.0, 1.0}), f"Неожиданные значения в {c}: {uniq}"

    print(f"Всего строк: {len(df)}, уникальных study: {df['study_id'].nunique()}")
    for c in LABEL_COLS:
        print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")

    # ---------- 2. Split по study_id ----------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_val_idx, test_idx = next(gss.split(df, groups=df["study_id"]))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test_df   = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, val_idx = next(gss2.split(train_val, groups=train_val["study_id"]))
    train_df = train_val.iloc[tr_idx].reset_index(drop=True)
    val_df   = train_val.iloc[val_idx].reset_index(drop=True)

    assert not (set(train_df.study_id) & set(val_df.study_id))
    assert not (set(train_df.study_id) & set(test_df.study_id))
    assert not (set(val_df.study_id)   & set(test_df.study_id))

    def describe(name, d):
        n_study = d["study_id"].nunique()
        print(f"{name:6s}: n={len(d):3d}, studies={n_study:3d}, "
              f"side={d['side'].value_counts().to_dict()}, "
              f"pos_rot={int(d['positioning_rotation'].sum(skipna=True))}/"
              f"{d['positioning_rotation'].notna().sum()}, "
              f"roi={int(d['roi_correctness'].sum(skipna=True))}/"
              f"{d['roi_correctness'].notna().sum()}")

    describe("train", train_df)
    describe("val  ", val_df)
    describe("test ", test_df)

    # ---------- 3. DataLoader ----------
    train_ds = HipQualityDataset(train_df, train=True)
    val_ds   = HipQualityDataset(val_df,   train=False)
    test_ds  = HipQualityDataset(test_df,  train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ---------- 4. Модель ----------
    model = HipQualityModel().to(DEVICE)
    pos_weight = compute_pos_weight(train_df, DEVICE)
    print(f"pos_weight: {pos_weight.cpu().numpy().round(2).tolist()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, side, y, m in train_loader:
            x, side, y, m = x.to(DEVICE), side.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(x, side)
                loss = masked_bce_loss(logits, y, m, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        scheduler.step()
        train_loss = running_loss / max(n_seen, 1)

        probs, ys, ms = evaluate(model, val_loader, DEVICE)
        val_f1s, thresholds, aucs = [], [], []
        for j, c in enumerate(LABEL_COLS):
            mask = ms[:, j] == 1
            y_true = ys[mask, j]
            y_prob = probs[mask, j]
            if len(y_true) == 0 or y_true.sum() == 0:
                val_f1s.append(0.0); thresholds.append(0.5); aucs.append(float("nan"))
                continue
            t, f1 = best_threshold(y_true, y_prob)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float("nan")
            val_f1s.append(f1); thresholds.append(t); aucs.append(auc)

        mean_f1 = float(np.nanmean(val_f1s))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:2d} | lr={lr_now:.2e} | train_loss={train_loss:.4f} | "
              f"val_F1={[round(x,3) for x in val_f1s]} | mean={mean_f1:.3f} | "
              f"AUC={[round(x,3) if not np.isnan(x) else 'nan' for x in aucs]}")

        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            torch.save({
                "model_state": model.state_dict(),
                "thresholds": thresholds,
                "val_f1": val_f1s,
                "val_auc": aucs,
                "epoch": epoch,
                "pos_weight": pos_weight.cpu().numpy().tolist(),
                "img_size": IMG_SIZE,
                "label_cols": LABEL_COLS,
            }, CKPT_PATH)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping на эпохе {epoch}")
                break

    print(f"\nBest val mean F1 = {best_val_f1:.3f}")

    # ---------- 5. Test ----------
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    thresholds = ckpt["thresholds"]
    print(f"Пороги из чекпоинта: {thresholds}")

    probs, ys, ms = evaluate(model, test_loader, DEVICE)

    print("\n=== TEST ===")
    for j, c in enumerate(LABEL_COLS):
        mask = ms[:, j] == 1
        y_true = ys[mask, j].astype(int)
        y_prob = probs[mask, j]
        if len(y_true) == 0:
            print(f"{c}: нет размеченных сэмплов в test")
            continue
        t = thresholds[j]
        y_pred = (y_prob >= t).astype(int)

        n_pos = int(y_true.sum()); n_neg = int((1 - y_true).sum())
        print(f"\n--- {c} ---")
        print(f"  n={len(y_true)}, pos={n_pos}, neg={n_neg}, threshold={t:.3f}")

        if n_pos == 0 or n_neg == 0:
            print("  одна из сторон пустая — метрики не считаем")
            continue

        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
            ap  = average_precision_score(y_true, y_prob)
        except ValueError:
            auc, ap = float("nan"), float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        f1_ci  = bootstrap_ci(y_true, y_pred,
                              lambda a, b: f1_score(a, b, zero_division=0))
        auc_ci = bootstrap_ci(y_true, y_prob, roc_auc_score)

        print(f"  F1        = {f1:.3f}  (95% CI {f1_ci[0]:.3f}..{f1_ci[1]:.3f})")
        print(f"  ROC-AUC   = {auc:.3f}  (95% CI {auc_ci[0]:.3f}..{auc_ci[1]:.3f})")
        print(f"  PR-AUC    = {ap:.3f}")
        print(f"  Sens      = {sens:.3f}")
        print(f"  Spec      = {spec:.3f}")
        print(f"  Confusion:\n{cm}")

    print("\nГотово. Чекпоинт:", CKPT_PATH.resolve())


if __name__ == "__main__":
    main()