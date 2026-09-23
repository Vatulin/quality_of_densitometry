"""Загрузка моделей и адаптер измерения отступов ROI для веб-сервиса."""

import importlib.util
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import expit
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dicom_minmax(dcm_path) -> np.ndarray:
    """min-max нормализация"""
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


# BODY PART MODEL

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


#  SPINE MODEL

class SpineModel:
    IMG_SIZE = 224
    FEATURE_VERSION = "spatial_resnet18_ridge_v2"

    def __init__(self, weights_path: str, debug: bool = False):
        self.device = DEVICE
        self.weights_path = Path(weights_path)
        self.debug = debug

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"[spine] веса не найдены: {self.weights_path}")

        ckpt = torch.load(
            self.weights_path, map_location=self.device, weights_only=False
        )

        # --- feature_version: жёсткая проверка, не warning ---
        self.feature_version = ckpt.get("feature_version")
        if self.feature_version != self.FEATURE_VERSION:
            raise ValueError(
                f"[spine] feature_version несовместим: "
                f"в чекпоинте {self.feature_version!r}, "
                f"ожидалось {self.FEATURE_VERSION!r}"
            )

        self.states = ckpt.get("estimators", [])
        if not self.states:
            raise ValueError(
                "[spine] в чекпоинте пустой список estimators — "
                "модель не сможет ничего предсказать"
            )

        self.threshold = float(ckpt.get("threshold", 0.5))
        if not np.isfinite(self.threshold):
            raise ValueError(f"[spine] некорректный threshold: {self.threshold!r}")

        self.uses_cnn = bool(ckpt.get("uses_cnn", False))
        self.encoder_kind = ckpt.get("encoder_kind", "resnet18")

        self.encoder = None
        if self.uses_cnn:
            enc_state = ckpt.get("encoder_state_dict", {})
            if not enc_state:
                raise ValueError(
                    "[spine] uses_cnn=True, но encoder_state_dict пуст: "
                    "веса энкодера не сохранены, предсказания будут шумом"
                )
            self.encoder = self._build_encoder(self.encoder_kind)
            missing, unexpected = self.encoder.load_state_dict(
                enc_state, strict=False
            )
            if self.debug and (missing or unexpected):
                print(f"[spine] encoder load: missing={missing}, "
                      f"unexpected={unexpected}")
            self.encoder.to(self.device).eval()

        print(
            f"✅ [spine] загружено: {self.weights_path} "
            f"(CNN={self.uses_cnn}, encoder={self.encoder_kind}, "
            f"threshold={self.threshold:.4f}, folds={len(self.states)})"
        )

    @staticmethod
    def _build_encoder(kind: str) -> nn.Module:
        if kind == "resnet18":
            model = models.resnet18(weights=None)
            encoder = nn.Sequential(
                *list(model.children())[:-2], nn.AdaptiveAvgPool2d((2, 2))
            )
        elif kind == "xrv-densenet121":
            model = models.densenet121(weights=None)
            model.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            encoder = nn.Sequential(
                model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2))
            )
        else:
            raise ValueError(f"Неизвестный экстрактор: {kind}")
        encoder.encoder_kind = kind
        encoder.requires_grad_(False)
        return encoder.eval()

    @staticmethod
    def _as_gray_unit(img) -> np.ndarray:
        if isinstance(img, Image.Image):
            if img.mode != "L":
                img = img.convert("L")
            arr = np.asarray(img, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"[spine] PIL дал shape={arr.shape}")
            return arr / 255.0

        if not isinstance(img, np.ndarray):
            raise TypeError(
                f"[spine] ожидался PIL.Image или np.ndarray, "
                f"получено {type(img).__name__}"
            )

        arr = img
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                # Rec.601 luma — тот же переход, что делает PIL.convert("L")
                arr = (
                    0.299 * arr[..., 0]
                    + 0.587 * arr[..., 1]
                    + 0.114 * arr[..., 2]
                )
            elif arr.shape[2] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"[spine] неподдерживаемый shape={arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"[spine] ожидался 2D, shape={arr.shape}")
        if arr.size == 0:
            raise ValueError("[spine] пустое изображение")

        arr = arr.astype(np.float64, copy=False)
        mn, mx = float(arr.min()), float(arr.max())
        if not np.isfinite(mn) or not np.isfinite(mx):
            raise ValueError("[spine] изображение содержит NaN/Inf")

        if mx <= 1.0 + 1e-6:
            return arr  # уже в [0,1]
        if mx <= 255.0 + 1e-6:
            return arr / 255.0  # 8-битный диапазон
        # 12/16-битный — min-max, как в train.read_image
        rng = mx - mn
        if rng <= 0:
            raise ValueError("[spine] постоянная яркость изображения")
        return (arr - mn) / rng

    def _geometry_features(self, img) -> np.ndarray:
        """Точная копия функции из train.py (spine_model)."""
        arr = self._as_gray_unit(img)
        features = []
        h, w = arr.shape
        for top, bottom in ((0.12, 0.72), (0.22, 0.82)):
            y0, y1, x0, x1 = (
                int(top * h), int(bottom * h), int(0.2 * w), int(0.8 * w)
            )
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
                            score[np.clip(previous, 0, width - 1)]
                            - 0.15 * delta ** 2,
                            -np.inf,
                        ))
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
                sampled = np.interp(
                    np.linspace(0, len(xx) - 1, 12),
                    np.arange(len(xx)), xx,
                )
                features.extend(np.abs(sampled - np.median(sampled)) / w)
                features.extend(np.abs(np.diff(sampled)) / w)
        result = np.asarray(features, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("[spine] геометрические признаки содержат NaN/Inf")
        return result

    @torch.inference_mode()
    def _cnn_features(self, img) -> np.ndarray:
        arr = self._as_gray_unit(img)  # уже [0,1], float64
        x = torch.from_numpy(arr).float()[None, None].to(self.device)
        if self.encoder_kind == "xrv-densenet121":
            x = (x * 2 - 1) * 1024
        else:
            x = x.repeat(1, 3, 1, 1)
            mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            x = (x - mean) / std
        spatial = (self.encoder(x) + self.encoder(x.flip(-1)).flip(-1)) / 2
        symmetric = torch.cat(
            (spatial.mean(-1), (spatial[..., 0] - spatial[..., 1]).abs()),
            dim=1,
        )
        return symmetric.flatten(1).cpu().numpy().astype(np.float64)

    def _predict_estimator(self, state, x: np.ndarray) -> np.ndarray:
        n_geometry = state["n_geometry"]

        def array(key):
            # .cpu() — страховка: тензор может быть на любом устройстве
            return state[key].cpu().numpy()

        if state["kind"] == "forest":
            x = np.asarray(x[:, :n_geometry], dtype=np.float32)
            predictions = []
            for tree in state["trees"]:
                left     = tree["left"].cpu().numpy()
                right    = tree["right"].cpu().numpy()
                feature  = tree["feature"].cpu().numpy()
                threshold = tree["threshold"].cpu().numpy()
                nodes = np.zeros(len(x), dtype=np.int64)
                while True:
                    active = np.flatnonzero(left[nodes] != -1)
                    if len(active) == 0:
                        break
                    current = nodes[active]
                    nodes[active] = np.where(
                        x[active, feature[current]] <= threshold[current],
                        left[current], right[current],
                    )
                predictions.append(tree["probability"].cpu().numpy()[nodes])
            return np.mean(predictions, axis=0)

        geom = (x[:, :n_geometry] - array("geometry_mean")) / array("geometry_scale")
        if state["features"] == "combined":
            cnn = (x[:, n_geometry:] - array("cnn_mean")) / array("cnn_scale")
            cnn = (cnn - array("pca_mean")) @ array("pca_components").T
            cnn = (cnn - array("pc_mean")) / array("pc_scale")
            geom = np.concatenate((geom, cnn), axis=1)

        # intercept мог быть сохранён как тензор — приводим к float
        intercept = state["intercept"]
        if isinstance(intercept, torch.Tensor):
            intercept = float(intercept.cpu().item())
        return expit(geom @ array("coef") + intercept)
    
    def _predict_ensemble(self, x: np.ndarray) -> np.ndarray:
        return np.mean(
            [self._predict_estimator(s, x) for s in self.states], axis=0
        )

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
            if not np.isfinite(prob):
                raise ValueError(f"[spine] предсказание не число: {prob!r}")

            class_id = 1 if prob >= self.threshold else 0
            result = {
                "status": "success",
                "class_id": class_id,
                "confidence": float(prob) if class_id == 1 else float(1.0 - prob),
                "probability": prob,
            }
            if self.debug:
                result["debug"] = {
                    "threshold": self.threshold,
                    "n_features": int(x.shape[1]),
                    "n_estimators": len(self.states),
                    "uses_cnn": self.uses_cnn,
                    "geom_min": float(np.min(geom)),
                    "geom_max": float(np.max(geom)),
                }
            return result

        except Exception as e:
            # Раньше ошибка молча уходила в {"status": "error"} и вызывающий
            # код её игнорировал — выглядело как «модель ничего не выдаёт».
            import traceback
            tb = traceback.format_exc()
            print(f"❌ [spine] ошибка на {dcm_path}:\n{tb}", flush=True)
            return {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb if self.debug else None,
            }


# HIP QUALITY (VERTEL) MODEL
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


# ARTIFACT MODEL 


class ArtifactModel:
    MODEL_NAME = "artifact"
    IMG_SIZE = 384
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, weights_path: str):
        self.device = DEVICE
        self.weights_path = Path(weights_path)
        try:
            import timm
        except ImportError as e:
            raise ImportError(f"Для {self.MODEL_NAME}_model нужен timm: pip install timm") from e

        self.model = timm.create_model("resnet18", pretrained=False, num_classes=1)
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.threshold = float(ckpt.get("best_threshold", 0.5))
        else:
            self.model.load_state_dict(ckpt)
            self.threshold = 0.5
        self.model.to(self.device).eval()
        print(f"✅ [{self.MODEL_NAME}] загружено: {self.weights_path} (threshold={self.threshold:.3f})")

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


class SpinePositionModel(ArtifactModel):
    """ResNet18 с предобработкой из spine_position_model/train.py."""

    MODEL_NAME = "spine_position"
    IMG_SIZE = 256

    @staticmethod
    def _read_dxa(dcm_path) -> np.ndarray:
        ds = pydicom.dcmread(str(dcm_path))
        img = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            img = img.max() - img
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min()) * 255.0
        else:
            img = np.zeros_like(img)
        return np.stack([img.astype(np.uint8)] * 3, axis=-1)


class PositionModel:
    """Адаптер общего CV-алгоритма: 30 мм сверху/снизу, 20 мм сбоку."""

    LABELS = {
        "top_margin_too_small": "Недостаточный верхний отступ ROI (менее 30 мм)",
        "bottom_margin_too_small": "Недостаточный нижний отступ ROI (менее 30 мм)",
        "lateral_margin_too_small": "Недостаточный боковой отступ ROI (менее 20 мм)",
        "incomplete_roi_at_bottom": "Область интереса неполна у нижней границы изображения",
        "top_margin_borderline": "Возможен дефект ROI: верхний отступ около порога 30 мм",
        "bottom_margin_borderline": "Возможен дефект ROI: нижний отступ около порога 30 мм",
        "lateral_margin_borderline": "Возможен дефект ROI: боковой отступ около порога 20 мм",
    }

    def __init__(self, module_path: str):
        spec = importlib.util.spec_from_file_location("_dxa_roi", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Не удалось загрузить алгоритм ROI: {module_path}")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def predict(self, dcm_path: str, side: str) -> Dict:
        try:
            report = self.module.analyze_roi(dcm_path, side)
            if report["processing_status"] != "Success":
                return {"status": "error", "error": report.get("error") or report["conclusion"],
                        "roi_details": report}
            return {
                "status": "success",
                "class_id": report["quality_class"],
                "violations": [self.LABELS.get(code, code) for code in report["violation_type"]],
                "roi_details": report,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
