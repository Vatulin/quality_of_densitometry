import uuid
import zipfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import expit
from torchvision import models

# --- НАСТРОЙКИ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIXEL_SPACING_X = 0.6   # мм (из ТЗ)
PIXEL_SPACING_Y = 1.05  # мм (из ТЗ)


# ====================== ОБЩИЕ УТИЛИТЫ ======================

def load_dicom_minmax(dcm_path) -> np.ndarray:
    """min-max нормализация — как во всех тренировочных скриптах."""
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def to_square_gray(arr: np.ndarray, size: int) -> Image.Image:
    """min-max → PIL L → паддинг до квадрата чёрным → bilinear resize."""
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.Resampling.BILINEAR)


# ====================== BODY PART MODEL ======================
# Архитектура из import numpy as np.txt: ResNet18, 2 класса (spine / hip),
# 1-канальный вход, IMG_SIZE=224.

class BodyPartModel:
    CLASS_NAMES = ["spine", "hip_right", "hip_left"]
    IMG_SIZE = 224

    def __init__(self, weights_path: str):
        self.device = DEVICE
        self.weights_path = Path(weights_path)

        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # Определяем число входных каналов первого conv
        in_channels = 1
        if "features.0.0.weight" in state:
            in_channels = int(state["features.0.0.weight"].shape[1])

        # Определяем число классов из classifier
        n_classes = len(self.CLASS_NAMES)
        for key in ("classifier.1.weight", "classifier.weight"):
            if key in state:
                n_classes = int(state[key].shape[0])
                break
        if n_classes != len(self.CLASS_NAMES):
            raise ValueError(f"Ожидали {len(self.CLASS_NAMES)} классов, в чекпоинте {n_classes}")

        model = models.efficientnet_b0(weights=None)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)

        # На случай сохранения через DataParallel
        first_key = next(iter(state))
        if first_key.startswith("model."):
            state = {k.replace("model.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        print(f"✅ [body_part] загружено: {self.weights_path} "
              f"(classes={self.CLASS_NAMES}, in_ch={in_channels})")

    def predict(self, dcm_path: str) -> Dict:
        try:
            arr = load_dicom_minmax(dcm_path)
            img = to_square_gray(arr, self.IMG_SIZE)
            # Как в first_model/test.py: только /255, без ImageNet-нормализации
            x = torch.from_numpy(np.array(img, copy=True)).float()[None, None] / 255.0
            x = x.to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            return {
                "status": "success",
                "class_id": pred_class,
                "class_name": self.CLASS_NAMES[pred_class],
                "confidence": float(probs.max()),
                "probabilities": probs.tolist(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ====================== SPINE MODEL ======================
# Из "Обучение на малом наборе…": ансамбль оценщиков
# (BalancedRandomForest или LogisticRegression) на геометрических признаках
# + (опционально) замороженные CNN-признаки ResNet18 / XRV-DenseNet121.

class SpineModel:
    IMG_SIZE = 224
    FEATURE_VERSION = "spatial_resnet18_ridge_v2"

    def __init__(self, weights_path: str):
        self.device = DEVICE
        self.weights_path = Path(weights_path)

        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        self.feature_version = ckpt.get("feature_version")
        if self.feature_version and self.feature_version != self.FEATURE_VERSION:
            print(f"⚠️ [spine] feature_version={self.feature_version} ≠ {self.FEATURE_VERSION}")
        self.states = ckpt.get("estimators", [])
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.uses_cnn = bool(ckpt.get("uses_cnn", False))
        self.encoder_kind = ckpt.get("encoder_kind", "resnet18")

        self.encoder = None
        if self.uses_cnn:
            self.encoder = self._build_encoder(self.encoder_kind)
            enc_state = ckpt.get("encoder_state_dict", {})
            if enc_state:
                self.encoder.load_state_dict(enc_state)
            self.encoder.to(self.device).eval()

        print(f"✅ [spine] загружено: {self.weights_path} "
              f"(CNN={self.uses_cnn}, encoder={self.encoder_kind}, threshold={self.threshold:.4f})")

    @staticmethod
    def _build_encoder(kind: str) -> nn.Module:
        if kind == "resnet18":
            model = models.resnet18(weights=None)
            encoder = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d((2, 2)))
        elif kind == "xrv-densenet121":
            model = models.densenet121(weights=None)
            model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            encoder = nn.Sequential(model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)))
        else:
            raise ValueError(f"Неизвестный экстрактор: {kind}")
        encoder.encoder_kind = kind
        encoder.requires_grad_(False)
        return encoder.eval()

    def _geometry_features(self, img: Image.Image) -> np.ndarray:
        """Точная копия функции из train.py (spine_model)."""
        arr = np.asarray(img, dtype=np.float64) / 255.0
        features = []
        h, w = arr.shape
        for top, bottom in ((0.12, 0.72), (0.22, 0.82)):
            y0, y1, x0, x1 = int(top * h), int(bottom * h), int(0.2 * w), int(0.8 * w)
            yy = np.arange(y0, y1, dtype=float)
            for sigma in (2.0, 5.0):
                smooth = gaussian_filter(arr, sigma=(2, sigma))
                background = gaussian_filter(arr, sigma=(2, 18))
                ridge = (smooth - background)[y0:y1, x0:x1]
                ridge /= max(float(np.std(ridge)), 1e-6)
                width = ridge.shape[1]
                xs = np.arange(width)
                score = ridge[0] - 0.1 * ((xs - width / 2) / (width / 2)) ** 2
                back = np.zeros(ridge.shape, dtype=np.int64)
                for row in range(1, len(ridge)):
                    options = []
                    for delta in range(-2, 3):
                        previous = xs + delta
                        valid = (previous >= 0) & (previous < width)
                        options.append(np.where(
                            valid,
                            score[np.clip(previous, 0, width - 1)] - 0.15 * delta ** 2,
                            -np.inf))
                    options = np.asarray(options)
                    best = options.argmax(axis=0)
                    back[row] = np.clip(xs + best - 2, 0, width - 1)
                    score = ridge[row] + options[best, xs]
                path = np.zeros(len(ridge), dtype=np.int64)
                path[-1] = score.argmax()
                for row in range(len(ridge) - 1, 0, -1):
                    path[row - 1] = back[row, path[row]]
                xx = median_filter(path.astype(float), size=7) + x0
                slope, intercept = np.polyfit(yy, xx, 1)
                residual = xx - (slope * yy + intercept)
                features.extend([
                    abs(np.degrees(np.arctan(slope))), np.std(residual) / w,
                    np.ptp(xx) / w, abs(xx[-1] - xx[0]) / len(xx),
                    np.mean((path == 0) | (path == width - 1)),
                    np.mean(ridge[np.arange(len(path)), path]),
                ])
                for section in np.array_split(np.arange(len(xx)), 3):
                    local = np.polyfit(yy[section], xx[section], 1)[0]
                    features.append(abs(np.degrees(np.arctan(local))))
                sampled = np.interp(np.linspace(0, len(xx) - 1, 12), np.arange(len(xx)), xx)
                features.extend(np.abs(sampled - np.median(sampled)) / w)
                features.extend(np.abs(np.diff(sampled)) / w)
        return np.asarray(features, dtype=np.float64)

    @torch.inference_mode()
    def _cnn_features(self, img: Image.Image) -> np.ndarray:
        x = torch.from_numpy(np.array(img, copy=True)).float()[None, None] / 255.0
        x = x.to(self.device)
        if self.encoder_kind == "xrv-densenet121":
            x = (x * 2 - 1) * 1024
        else:
            x = x.repeat(1, 3, 1, 1)
            mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            x = (x - mean) / std
        spatial = (self.encoder(x) + self.encoder(x.flip(-1)).flip(-1)) / 2
        symmetric = torch.cat((spatial.mean(-1), (spatial[..., 0] - spatial[..., 1]).abs()), dim=1)
        return symmetric.flatten(1).cpu().numpy().astype(np.float64)

    def _predict_estimator(self, state, x: np.ndarray) -> np.ndarray:
        n_geometry = state["n_geometry"]

        def array(key):
            return state[key].numpy()

        if state["kind"] == "forest":
            x = np.asarray(x[:, :n_geometry], dtype=np.float32)
            predictions = []
            for tree in state["trees"]:
                left, right = tree["left"].numpy(), tree["right"].numpy()
                feature, threshold = tree["feature"].numpy(), tree["threshold"].numpy()
                nodes = np.zeros(len(x), dtype=np.int64)
                while True:
                    active = np.flatnonzero(left[nodes] != -1)
                    if len(active) == 0:
                        break
                    current = nodes[active]
                    nodes[active] = np.where(
                        x[active, feature[current]] <= threshold[current],
                        left[current], right[current])
                predictions.append(tree["probability"].numpy()[nodes])
            return np.mean(predictions, axis=0)

        # LogisticRegression + масштабирование/PCA
        geom = (x[:, :n_geometry] - array("geometry_mean")) / array("geometry_scale")
        if state["features"] == "combined":
            cnn = (x[:, n_geometry:] - array("cnn_mean")) / array("cnn_scale")
            cnn = (cnn - array("pca_mean")) @ array("pca_components").T
            cnn = (cnn - array("pc_mean")) / array("pc_scale")
            geom = np.concatenate((geom, cnn), axis=1)
        return expit(geom @ array("coef") + state["intercept"])

    def _predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        return np.mean([self._predict_estimator(s, x) for s in self.states], axis=0)

    def predict(self, dcm_path: str) -> Dict:
        try:
            arr = load_dicom_minmax(dcm_path)
            img = to_square_gray(arr, self.IMG_SIZE)
            geom = self._geometry_features(img)[None, :]
            if self.uses_cnn and self.encoder is not None:
                cnn = self._cnn_features(img)
                x = np.concatenate((geom, cnn), axis=1)
            else:
                x = geom
            prob = float(self._predict_ensemble(x)[0])
            class_id = 1 if prob >= self.threshold else 0
            return {
                "status": "success",
                "class_id": class_id,
                "confidence": float(prob) if class_id == 1 else float(1.0 - prob),
                "probability": float(prob),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ====================== HIP QUALITY (VERTEL) MODEL ======================
# Из Pasted text.py: EfficientNet_B0 + side-embedding, 2 выхода
# (positioning_rotation, roi_correctness), IMG_SIZE=384.

class _HipQualityNet(nn.Module):
    LABEL_COUNT = 2

    def __init__(self, n_side=2, side_dim=32):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.side_emb = nn.Embedding(n_side, side_dim)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(feat_dim + side_dim, self.LABEL_COUNT)

    def forward(self, x, side):
        feats = self.backbone(x)
        side_f = self.side_emb(side)
        h = torch.cat([feats, side_f], dim=1)
        h = self.dropout(h)
        return self.head(h)


class HipQualityModel:
    IMG_SIZE = 384
    LABEL_COLS = ["positioning_rotation", "roi_correctness"]
    LABEL_RU = {
        "positioning_rotation": "Не выравнена ось / ротация",
        "roi_correctness": "Некорректная область интереса",
    }

    def __init__(self, weights_path: str):
        self.device = DEVICE
        self.weights_path = Path(weights_path)
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)

        state = None
        if isinstance(ckpt, dict):
            for k in ("model_state", "model_state_dict", "state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict) and len(ckpt[k]) > 0:
                    state = ckpt[k]
                    break
        if state is None:
            state = ckpt

        self.model = _HipQualityNet().to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.thresholds = ckpt.get("thresholds", [0.5, 0.5]) if isinstance(ckpt, dict) else [0.5, 0.5]
        self.img_size = int(ckpt.get("img_size", 384)) if isinstance(ckpt, dict) else 384
        print(f"✅ [hip_quality] загружено: {self.weights_path} "
              f"(thresholds={self.thresholds}, img_size={self.img_size})")

    def predict(self, dcm_path: str, side: str | None = None) -> Dict:
        """
        side: "right" / "left" / None.
        В тренировке side_id = (side == "right").astype(int) → right=0, left=1.
        Если сторона неизвестна — считаем обе и берём максимум (страховка).
        """
        try:
            arr = load_dicom_minmax(dcm_path)
            img = to_square_gray(arr, self.img_size)
            x = torch.from_numpy(np.array(img, copy=True)).float()[None, None] / 255.0
            x = ((x - 0.5) / 0.25).to(self.device)

            if side == "right":
                side_ids = [0]
            elif side == "left":
                side_ids = [1]
            else:
                side_ids = [0, 1]

            probs_per_side = []
            with torch.no_grad():
                for sv in side_ids:
                    side_t = torch.tensor([sv], dtype=torch.long, device=self.device)
                    logits = self.model(x, side_t)
                    probs_per_side.append(torch.sigmoid(logits).cpu().numpy()[0])

            probs = (np.max(np.stack(probs_per_side, axis=0), axis=0)
                    if len(probs_per_side) > 1 else probs_per_side[0])

            violations, confidences = [], []
            for j, name in enumerate(self.LABEL_COLS):
                t = self.thresholds[j] if j < len(self.thresholds) else 0.5
                if probs[j] >= t:
                    violations.append(self.LABEL_RU.get(name, name))
                    confidences.append(float(probs[j]))

            class_id = 1 if violations else 0
            confidence = max(confidences) if confidences else float(1.0 - probs.max())
            return {
                "status": "success",
                "class_id": class_id,
                "confidence": confidence,
                "violations": violations,
                "probabilities": probs.tolist(),
                "label_cols": self.LABEL_COLS,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ====================== ARTIFACT MODEL ======================
# Из import os.txt (второй файл): timm.resnet18, 1 выход, sigmoid, порог.
# Препроцессинг: percentile(1,99) clip → CLAHE → 3 канала → ImageNet normalize.

class ArtifactModel:
    IMG_SIZE = 384
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights_path: str):
        self.device = DEVICE
        self.weights_path = Path(weights_path)
        try:
            import timm
        except ImportError as e:
            raise ImportError("Для artifact_model нужен timm: pip install timm") from e

        self.model = timm.create_model("resnet18", pretrained=False, num_classes=1)
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.threshold = float(ckpt.get("best_threshold", 0.5))
        else:
            self.model.load_state_dict(ckpt)
            self.threshold = 0.5
        self.model.to(self.device).eval()
        print(f"✅ [artifact] загружено: {self.weights_path} (threshold={self.threshold:.3f})")

    @staticmethod
    def _read_dxa(dcm_path) -> np.ndarray:
        ds = pydicom.dcmread(str(dcm_path))
        img = ds.pixel_array.astype(np.float32)
        p_min, p_max = np.percentile(img, (1, 99))
        img = np.clip(img, p_min, p_max)
        img = (img - p_min) / (p_max - p_min + 1e-8)
        img_uint8 = (img * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        return np.stack([img_clahe] * 3, axis=-1)

    def predict(self, dcm_path: str) -> Dict:
        try:
            img = self._read_dxa(dcm_path)
            img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - self.MEAN) / self.STD
            x = torch.from_numpy(img.transpose(2, 0, 1)).float()[None].to(self.device)
            with torch.no_grad():
                logit = self.model(x).squeeze(-1)
                prob = float(torch.sigmoid(logit).cpu().item())
            class_id = 1 if prob >= self.threshold else 0
            return {
                "status": "success",
                "class_id": class_id,
                "confidence": prob if class_id == 1 else 1.0 - prob,
                "probability": prob,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ====================== POSITION MODEL ======================
# Классический CV-алгоритм (без весов) — проверка грубых отклонений.

class PositionModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        print("✅ [position] инициализирована (CV алгоритм)")

    def predict(self, dcm_path: str, body_part: str = "hip") -> Dict:
        try:
            arr = load_dicom_minmax(dcm_path)
            arr_u8 = (arr * 255).astype(np.uint8)
            violations = []
            confidence = 0.0
            smooth = cv2.GaussianBlur(arr_u8, (3, 3), 0)
            threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = smooth.shape

            bottom_region = smooth[int(h * 0.75):, :]
            if (bottom_region > threshold * 0.1).sum() < 100:
                violations.append("Некорректная область интереса")
                confidence = max(confidence, 0.75)

            top_region = smooth[:int(h * 0.25), :]
            if (top_region > threshold * 0.1).sum() < 100:
                violations.append("Некорректная укладка")
                confidence = max(confidence, 0.70)

            return {
                "status": "success",
                "class_id": 1 if violations else 0,
                "confidence": confidence,
                "violations": violations,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ====================== АНАЛИЗАТОР ======================

class MultiModelAnalyzer:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, object] = {}
        self._load_models()

    def _try_load(self, name: str, cls, *paths) -> bool:
        for p in paths:
            p = Path(p)
            if p.exists():
                try:
                    self.models[name] = cls(str(p))
                    return True
                except Exception as e:
                    print(f"⚠️ [{name}] не удалось загрузить {p}: {e}")
        print(f"⚠️ [{name}] веса не найдены: {[str(p) for p in paths]}")
        return False

    def _load_models(self):
        md = self.models_dir

        # 1) Body part — ResNet18, 2 класса (spine / hip)
        self._try_load(
            "body_part", BodyPartModel,
            md / "body_part_model" / "weights" / "best_model.pt",
            md / "body_part_model" / "best_model.pt",
        )

        # 2) Spine — ансамбль (RF или LogReg) + опционально CNN
        self._try_load(
            "spine", SpineModel,
            md / "spine_model" / "weights" / "best_model.pt",
            md / "spine_model" / "best_model.pt",
        )

        # 3) Hip quality (vertel) — EfficientNet_B0 + side-embedding
        self._try_load(
            "hip_quality", HipQualityModel,
            md / "vertel_model" / "weights" / "best_hip_quality.pt",
            md / "vertel_model" / "weights" / "best_model.pt",
            md / "vertel_model" / "best_hip_quality.pt",
            md / "vertel_model" / "best_model.pt",
        )

        # 4) Artifacts — timm.resnet18, sigmoid
        self._try_load(
            "artifact", ArtifactModel,
            md / "artifact_model" / "weights" / "best_artifact_model.pth",
            md / "artifact_model" / "weights" / "best_model.pt",
            md / "artifact_model" / "best_artifact_model.pth",
        )

        # 5) Position — CV, без весов
        pos_path = md / "position_model"
        if pos_path.exists():
            self.models["position"] = PositionModel(str(pos_path))

        print(f"✅ Всего успешно загружено моделей: {len(self.models)}")

    def analyze(self, dcm_path: str) -> Dict:
        results = {
            "filename": Path(dcm_path).name,
            "anatomical_region": None,
            "hip_side": None,
            "quality_class": 0,
            "quality_prob": 0.0,
            "violation_prob": 0.0,     # ← сырая вероятность нарушения (для отладки/UI)
            "violation_type": [],
            "model_predictions": {},
        }

        # 1) Часть тела и сторона
        hip_side = None
        if "body_part" in self.models:
            pred = self.models["body_part"].predict(dcm_path)
            results["model_predictions"]["body_part"] = pred
            if pred["status"] == "success":
                name = pred.get("class_name")
                if name == "spine":
                    results["anatomical_region"] = "Поясничный отдел позвоночника"
                elif name == "hip_right":
                    results["anatomical_region"] = "Проксимальный отдел бедра"
                    hip_side = "right"
                elif name == "hip_left":
                    results["anatomical_region"] = "Проксимальный отдел бедра"
                    hip_side = "left"
        results["hip_side"] = hip_side

        region = results["anatomical_region"]
        violations: List[str] = []
        violation_probs: List[float] = []   # ← все сырые вероятности нарушений

        # 2) Позвоночник
        if region == "Поясничный отдел позвоночника" and "spine" in self.models:
            pred = self.models["spine"].predict(dcm_path)
            results["model_predictions"]["spine"] = pred
            if pred["status"] == "success":
                violation_probs.append(float(pred.get("probability", 0.0)))
                if pred["class_id"] == 1:
                    violations.append("Не выравнена ось позвоночника")

        # 3) Бедро
        if region == "Проксимальный отдел бедра" and "hip_quality" in self.models:
            pred = self.models["hip_quality"].predict(dcm_path, side=hip_side)
            results["model_predictions"]["hip_quality"] = pred
            if pred["status"] == "success":
                for p in pred.get("probabilities", []):
                    violation_probs.append(float(p))
                if pred["class_id"] == 1:
                    for v in pred.get("violations", []):
                        if v not in violations:
                            violations.append(v)

        # 4) Артефакты — всегда
        if "artifact" in self.models:
            pred = self.models["artifact"].predict(dcm_path)
            results["model_predictions"]["artifact"] = pred
            if pred["status"] == "success":
                violation_probs.append(float(pred.get("probability", 0.0)))
                if pred["class_id"] > 0 and "Некорректная укладка" not in violations:
                    violations.append("Некорректная укладка")

        # 5) Position (CV) — только бедро; вероятности не даёт, только флаг
        if "position" in self.models and region == "Проксимальный отдел бедра":
            pred = self.models["position"].predict(dcm_path, body_part="hip")
            results["model_predictions"]["position"] = pred
            if pred["status"] == "success" and pred["class_id"] == 1:
                for v in pred.get("violations", []):
                    if v not in violations:
                        violations.append(v)

        max_violation_prob = max(violation_probs) if violation_probs else 0.0

        results["violation_type"] = violations
        results["violation_prob"] = float(max_violation_prob)

        if violations:
            results["quality_class"] = 1
            results["quality_prob"] = float(max_violation_prob)
        else:
            results["quality_class"] = 0
            results["quality_prob"] = float(1.0 - max_violation_prob)  # ← уверенность «всё ОК»

        return results


# ====================== FASTAPI ======================

app = FastAPI(title="DXA Multi-Model Analyzer")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analyzer: MultiModelAnalyzer | None = None
tasks: Dict = {}


@app.on_event("startup")
async def startup_event():
    global analyzer
    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        print(f"✅ Папка моделей найдена: {models_dir}")
        analyzer = MultiModelAnalyzer(models_dir)
    else:
        print(f"⚠️ Папка models не найдена по пути: {models_dir}")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/analyze")
async def analyze_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "Файл не выбран"}
    task_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    tasks[task_id] = {"status": "PROCESSING", "progress": 0, "total": 1, "results": []}
    if file.filename.lower().endswith(".zip"):
        background_tasks.add_task(process_zip_with_models, task_id, file_path)
    else:
        background_tasks.add_task(process_single_file, task_id, file_path)
    return {"task_id": task_id}


def process_single_file(task_id: str, file_path: Path):
    try:
        if analyzer is None:
            raise Exception("Анализатор не инициализирован")
        result = analyzer.analyze(str(file_path))
        tasks[task_id]["results"] = [result]
        tasks[task_id]["progress"] = 1
        tasks[task_id]["total"] = 1
        tasks[task_id]["status"] = "COMPLETED"
    except Exception as e:
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)


def process_zip_with_models(task_id: str, zip_path: Path):
    try:
        if analyzer is None:
            raise Exception("Анализатор не инициализирован")
        extract_dir = UPLOAD_DIR / task_id
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        dcm_files = list(extract_dir.rglob("*.dcm")) + list(extract_dir.rglob("*.dicom"))
        tasks[task_id]["total"] = len(dcm_files)
        results = []
        for i, dcm_file in enumerate(dcm_files):
            result = analyzer.analyze(str(dcm_file))
            result["path"] = dcm_file.relative_to(extract_dir).as_posix()
            results.append(result)
            tasks[task_id]["progress"] = i + 1
        tasks[task_id]["results"] = results
        tasks[task_id]["status"] = "COMPLETED"
    except Exception as e:
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"error": "Задача не найдена"})


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "COMPLETED":
        return {"error": "Результат еще не готов"}

    side_ru = {"right": "Правый", "left": "Левый"}
    rows = []
    for result in task["results"]:
        rows.append({
            "filename": result["filename"],
            "path": result.get("path", ""),
            "anatomical_region": result["anatomical_region"] or "",
            "hip_side": side_ru.get(result.get("hip_side"), ""),
            "quality_class": result["quality_class"],
            "quality_prob": round(result["quality_prob"], 4),
            "violation_prob": round(result.get("violation_prob", 0.0), 4),
            "violation_type": ";".join(result["violation_type"]) if result["violation_type"] else "",
        })
    df = pd.DataFrame(rows)
    csv_path = UPLOAD_DIR / f"{task_id}_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return FileResponse(csv_path, filename=f"results_{task_id}.csv", media_type="text/csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)