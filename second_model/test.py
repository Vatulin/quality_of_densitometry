import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import models


# =============================================================
#                      КОНФИГ
# =============================================================
BODY_CKPT = Path(r"..\weights\best_model.pt")             # 1-я модель (spine/hip_r/hip_l)
HIP_CKPT  = Path(r"..\weights\best_hip_quality.pt")       # 2-я модель (quality)

BODY_IMG_SIZE = 288
HIP_IMG_SIZE  = 384
BODY_CLASSES  = ["spine", "hip_right", "hip_left"]
LABEL_COLS    = ["positioning_rotation", "roi_correctness"]

# Человекочитаемые названия
RU_REGION = {
    "spine":     "позвоночник",
    "hip_right": "правое бедро",
    "hip_left":  "левое бедро",
}
RU_LABEL = {
    "positioning_rotation": "позиционирование / ротация",
    "roi_correctness":      "область интересов (ROI)",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
#                   МОДЕЛЬ 1 — область
# =============================================================
class BodyPartModel(nn.Module):
    """Копия архитектуры первой модели. Без обёртки net., иначе ключи не совпадут."""
    def __init__(self, n_classes=3):
        super().__init__()
        m = models.efficientnet_b0(weights=None)
        old_conv = m.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        m.features[0][0] = new_conv
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)

        self.features = m.features
        self.avgpool = m.avgpool
        self.classifier = m.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# =============================================================
#                   МОДЕЛЬ 2 — качество
# =============================================================
class HipQualityModel(nn.Module):
    def __init__(self, n_side=2, side_dim=32, n_labels=2):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        backbone.features[0][0] = new_conv
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.side_emb = nn.Embedding(n_side, side_dim)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(feat_dim + side_dim, n_labels)

    def forward(self, x, side):
        feats = self.backbone(x)
        side_f = self.side_emb(side)
        h = torch.cat([feats, side_f], dim=1)
        return self.head(self.dropout(h))


# =============================================================
#                    ПРЕДОБРАБОТКА
# =============================================================
def load_dicom_image(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def prep_for_body(img, img_size=BODY_IMG_SIZE):
    img = img.resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0
    return x.unsqueeze(0)


def prep_for_hip(img, img_size=HIP_IMG_SIZE):
    img = img.resize((img_size, img_size), Image.BILINEAR)
    x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0
    x = (x - 0.5) / 0.25
    return x.unsqueeze(0)


# =============================================================
#                    ЕДИНЫЙ ПАЙПЛАЙН
# =============================================================
class DXAPipeline:
    def __init__(self, body_ckpt=BODY_CKPT, hip_ckpt=HIP_CKPT, device=None, verbose=True):
        self.device = device or DEVICE
        self.verbose = verbose

        if verbose:
            print(f"Устройство: {self.device}")

        # --- Модель 1 ---
        self.body_model = BodyPartModel(n_classes=len(BODY_CLASSES)).to(self.device)
        state = torch.load(body_ckpt, map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        # на случай чекпоинта с префиксом net.
        state = {k.replace("net.", "", 1): v for k, v in state.items()}
        self.body_model.load_state_dict(state)
        self.body_model.eval()
        if verbose:
            print(f"Модель областей:  загружена ({body_ckpt})")

        # --- Модель 2 ---
        ckpt = torch.load(hip_ckpt, map_location=self.device)
        self.hip_thresholds = ckpt["thresholds"]
        self.hip_img_size = ckpt.get("img_size", HIP_IMG_SIZE)
        self.label_cols = ckpt.get("label_cols", LABEL_COLS)

        self.hip_model = HipQualityModel(n_labels=len(self.label_cols)).to(self.device)
        self.hip_model.load_state_dict(ckpt["model_state"])
        self.hip_model.eval()
        if verbose:
            print(f"Модель качества:  загружена ({hip_ckpt})")
            print(f"Пороги: {[round(t, 3) for t in self.hip_thresholds]}")
        if verbose:
            print()

    # ---------- 1-я модель ----------
    @torch.no_grad()
    def predict_body(self, x):
        x = x.to(self.device)
        with autocast(enabled=(self.device == "cuda")):
            logits = self.body_model(x)
        probs = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
        cls_id = int(probs.argmax())
        return BODY_CLASSES[cls_id], float(probs[cls_id]), probs

    # ---------- 2-я модель ----------
    @torch.no_grad()
    def predict_hip_quality(self, x, side_id):
        x = x.to(self.device)
        side = torch.tensor([side_id], dtype=torch.long, device=self.device)
        with autocast(enabled=(self.device == "cuda")):
            logits = self.hip_model(x, side)
        return torch.sigmoid(logits).float().cpu().numpy()[0]

    # ---------- Полный сценарий на один файл ----------
    def predict_file(self, dcm_path, study_id=None, series_id=None):
        img = load_dicom_image(dcm_path)

        x_body = prep_for_body(img)
        body_cls, body_conf, _ = self.predict_body(x_body)

        result = {
            "file": str(dcm_path),
            "study_id": study_id or "",
            "series_id": series_id or "",
            "anatomical_region": body_cls,
            "body_confidence": round(body_conf, 4),
            "quality_class": 0,
            "violation_type": "none",
            "processing_status": "Success",
        }

        # Не бедро → выходим
        if body_cls == "spine":
            result["note"] = "для позвоночника модель качества не обучена"
            return result

        side = "right" if body_cls == "hip_right" else "left"
        side_id = 1 if side == "right" else 0
        result["side"] = side

        # Качество укладки
        try:
            x_hip = prep_for_hip(img, self.hip_img_size)
            probs = self.predict_hip_quality(x_hip, side_id)
        except Exception as e:
            result["processing_status"] = "Failure"
            result["error"] = str(e)
            return result

        violations = []
        for j, c in enumerate(self.label_cols):
            p = float(probs[j])
            t = float(self.hip_thresholds[j])
            flag = int(p >= t)
            result[f"prob_{c}"] = round(p, 4)
            result[f"pred_{c}"] = flag
            result[f"threshold_{c}"] = round(t, 3)
            if flag == 1:
                prefix = "right_hip" if side == "right" else "left_hip"
                violations.append(f"{prefix}_{c}")

        result["quality_class"] = int(len(violations) > 0)
        result["violation_type"] = ";".join(violations) if violations else "none"
        return result


# =============================================================
#                ЧЕЛОВЕКОЧИТАЕМЫЙ ВЫВОД
# =============================================================
def print_single_result(r):
    """Печатает результат по одному файлу понятным русским языком."""
    line = "═" * 60
    print(line)
    print(f"  Файл: {Path(r['file']).name}")
    print(line)

    region = r.get("anatomical_region", "unknown")
    conf = r.get("body_confidence", 0)
    print(f"  Часть тела:   {RU_REGION.get(region, region)}  "
          f"(уверенность {conf * 100:.1f}%)")

    if region == "spine":
        print(f"  Качество:     не оценивается")
        print(f"  Причина:      {r.get('note', 'модель обучена только на бедре')}")
        print(line)
        return

    side = r.get("side", "?")
    print(f"  Сторона:      {'правая' if side == 'right' else 'левая'}")

    qc = r.get("quality_class", 0)
    if qc == 1:
        print(f"  Качество:     НАРУШЕНИЕ")
    elif qc == 0:
        print(f"  Качество:     норма")
    else:
        print(f"  Качество:     ошибка обработки")
        if "error" in r:
            print(f"  Ошибка:       {r['error']}")
        print(line)
        return

    print(f"\n  Детали проверки:")
    for c in LABEL_COLS:
        p = r.get(f"prob_{c}")
        t = r.get(f"threshold_{c}")
        pred = r.get(f"pred_{c}")
        if p is None:
            continue
        status = "НАРУШЕНО" if pred == 1 else "норма   "
        label = RU_LABEL.get(c, c)
        print(f"    • {label:35s} {status}  "
              f"(вероятность {p*100:5.1f}%, порог {t*100:.0f}%)")

    print(f"\n  Тип нарушения: {r.get('violation_type', 'none')}")
    print(line)


def print_batch_summary(df):
    """Короткая сводка по обработанной папке."""
    line = "═" * 60
    print(line)
    print(f"  ИТОГИ ОБРАБОТКИ")
    print(line)

    total = len(df)
    ok = int((df["processing_status"] == "Success").sum()) if "processing_status" in df else total
    err = total - ok
    print(f"  Всего файлов:  {total}")
    print(f"  Успешно:       {ok}")
    if err:
        print(f"  С ошибками:    {err}")

    if "anatomical_region" in df.columns:
        print(f"\n  Распределение по частям тела:")
        counts = df["anatomical_region"].value_counts()
        for k, v in counts.items():
            name = RU_REGION.get(k, k)
            print(f"    {name:20s} {v:4d}")

    if "quality_class" in df.columns:
        hips = df[df["anatomical_region"].isin(["hip_right", "hip_left"])]
        if len(hips):
            n_ok  = int((hips["quality_class"] == 0).sum())
            n_bad = int((hips["quality_class"] == 1).sum())
            print(f"\n  Качество укладки бедра (всего {len(hips)}):")
            print(f"    норма        {n_ok:4d}")
            print(f"    нарушение    {n_bad:4d}")

            # разбивка по типам нарушений
            vt = hips[hips["violation_type"] != "none"]["violation_type"]
            if len(vt):
                print(f"\n  Типы нарушений:")
                # разворачиваем многострочные значения (a;b) в отдельные строки
                exploded = vt.str.split(";").explode().value_counts()
                for k, v in exploded.items():
                    # переводим код в человекочитаемое
                    side, _, label = k.partition("_hip_")
                    side_ru = "правое" if side == "right" else "левое"
                    label_ru = RU_LABEL.get(label, label)
                    print(f"    {side_ru} бедро — {label_ru}: {v}")

    print(line)


# =============================================================
#                       BATCH
# =============================================================
def find_dicoms(folder):
    files = []
    for p in Path(folder).rglob("*"):
        if not p.is_file():
            continue
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
            files.append(p)
        except Exception:
            continue
    return sorted(files)


def batch_predict(pipeline, input_dir, xlsx_out=None):
    files = find_dicoms(input_dir)
    print(f"Найдено DICOM-файлов: {len(files)}\n")

    rows = []
    for i, f in enumerate(files, 1):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            study_uid  = str(getattr(ds, "StudyInstanceUID", ""))
            series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            r = pipeline.predict_file(f, study_id=study_uid, series_id=series_uid)
        except Exception as e:
            r = {
                "file": str(f),
                "study_id": "", "series_id": "",
                "anatomical_region": "unknown",
                "quality_class": -1,
                "violation_type": f"error: {e}",
                "processing_status": "Failure",
                "error": str(e),
            }
        rows.append(r)
        if i % 20 == 0:
            print(f"  обработано {i}/{len(files)}...")

    df = pd.DataFrame(rows)

    preferred = [
        "file", "study_id", "series_id", "anatomical_region", "side",
        "quality_class", "violation_type", "processing_status",
        "body_confidence",
        "prob_positioning_rotation", "pred_positioning_rotation",
        "prob_roi_correctness", "pred_roi_correctness",
    ]
    cols = [c for c in preferred if c in df.columns] + \
           [c for c in df.columns if c not in preferred]
    df = df[cols]

    if xlsx_out:
        Path(xlsx_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(xlsx_out, index=False)
        print(f"\nРезультат сохранён в: {xlsx_out}")

    return df


# =============================================================
#                       CLI
# =============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Определение области и оценка качества укладки DXA."
    )
    ap.add_argument("path", help="DICOM-файл или папка с DICOM")
    ap.add_argument("--body-ckpt", default=str(BODY_CKPT),
                    help="чекпоинт модели областей")
    ap.add_argument("--hip-ckpt", default=str(HIP_CKPT),
                    help="чекпоинт модели качества")
    ap.add_argument("--xlsx", default=None,
                    help="сохранить результат батча в Excel")
    ap.add_argument("--json", action="store_true",
                    help="вывести машинный JSON вместо текста (для одного файла)")
    args = ap.parse_args()

    pipeline = DXAPipeline(args.body_ckpt, args.hip_ckpt)
    p = Path(args.path)

    if p.is_file():
        r = pipeline.predict_file(p)
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2))
        else:
            print_single_result(r)
    else:
        df = batch_predict(pipeline, p, xlsx_out=args.xlsx)
        print()
        print_batch_summary(df)


if __name__ == "__main__":
    main()