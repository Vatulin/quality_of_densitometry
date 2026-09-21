"""Обучение на малом наборе: defects=0 (<=5°), defects=1 (>5°).

Запуск: python train.py --device cpu
Медицинские признаки: python train.py --encoder xrv-densenet121
Для этого режима: python -m pip install torchxrayvision
Зависимости: torch torchvision numpy pandas pillow pydicom scipy scikit-learn imbalanced-learn
Замороженная RGB ResNet18 + признаки оси; CNN не дообучается на 6 снимках.
Сравниваются LogisticRegression + RandomOverSampler и BalancedRandomForest.
Вес класса 1 подбирается среди 1, 2, 4; выбор модели/порога по F2 при
specificity >= 0.75 (параметры --positive-weights, --beta, --min-specificity).
Медицинская DenseNet предобучена на chest X-ray, не DXA: преимущество не гарантировано.
Экстракторы заморожены; на разметке обучается классификатор признаков.
Выбор модели/порога: повторная групповая CV на development, без test.
Предобработка, PCA и балансировка обучаются строго внутри train каждого фолда.
Сохраняется ансамбль фолдов: weights/best_model.pt в папке spine_model.
Другой путь сохранения можно задать через --output.
Первый запуск может скачать ImageNet-веса в стандартный кеш torchvision.
Все промежуточные данные остаются в RAM. --epochs/--lr больше не нужны.

Метка defects должна означать именно наклон >5°, не любой дефект.
Геометрические признаки — эвристика, а не измерение клинического угла.
10 положительных примеров недостаточно для надёжной оценки обобщения.
OOF используется для выбора модели и порога: это не независимая оценка.
Прежний test сохраняется (при том же CSV и seed); он уже был просмотрен,
поэтому для окончательной оценки нужен новый внешний набор.

https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets?hl=ru
https://imbalanced-learn.org/stable/common_pitfalls.html
"""

import argparse
import random
from pathlib import Path
import imblearn
import sklearn

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DEFAULT_WEIGHTS = HERE / "weights" / "best_model.pt"
IMG_SIZE = 224
PREPROCESS = "dicom_minmax_square_224_v1"
FEATURE_VERSION = "spatial_resnet18_ridge_v2"
# Подтверждено пользователем: CR000004 имеет наклон >5°. Его pixel_array
# побитово совпадает с CR000001 из CSV, ошибочно помеченным 0.
# Исправление применяется только к обучающей разметке, не к предсказанию.
LABEL_CORRECTIONS = {
    "Data/Исследования/2.25.102755089973625799055786462646268820450/"
    "series_002_2_CR/A2504257770 DXA/CR DXA/CR000001.dcm": 1,
}


def build_model(pretrained=False):
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    rgb_weights = model.conv1.weight.detach().clone()
    model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(rgb_weights.sum(dim=1, keepdim=True))
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def read_image(path):
    """Та же min-max обработка, что у first_model; геометрию не искажаем."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError(f"Ожидался конечный двумерный grayscale DICOM: {path}")
    arr -= arr.min()
    if arr.max() == 0:
        raise ValueError(f"Изображение имеет постоянную яркость: {path}")
    arr /= arr.max()
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    side = max(img.size)
    square = Image.new("L", (side, side), 0)
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return square.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)


def image_tensor(img):
    return torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0


def read_dataset(csv_path, root):
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding=encoding,
                             dtype={"study_id": str, "rel_path": str})
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Не удалось прочитать CSV как UTF-8 или Windows-1251")
    required = {"rel_path", "defects", "study_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV должен содержать {sorted(required)}")
    if df[list(required)].isna().any().any():
        raise ValueError("Пустые пути, study_id или метки в CSV")
    if not df.defects.isin([0, 1]).all():
        raise ValueError("defects должен содержать только 0 и 1")
    if "body_spine" in df and not df.body_spine.eq(1).all():
        raise ValueError("CSV содержит снимки, не помеченные как позвоночник")
    df["defects"] = df.defects.astype(int)
    # Страты старого разбиения сохраняем для сопоставимости с прежним test.
    df["split_label"] = df.defects.copy()
    for rel_path, corrected_label in LABEL_CORRECTIONS.items():
        mask = df.rel_path.str.replace("\\", "/", regex=False).eq(rel_path)
        df.loc[mask, "defects"] = corrected_label
    df["path"] = [str((root / str(p).replace("\\", "/")).resolve()) for p in df.rel_path]
    if df.path.duplicated().any():
        raise ValueError("В CSV есть повторяющиеся пути; устраните дубликаты")
    missing = [p for p in df.path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Не найдены {len(missing)} файлов; пример: {missing[0]}")
    return df


def split_dataset(df, seed):
    """60/20/20 по исследованиям; при смешанных метках страта = наличие дефекта."""
    split_column = "split_label" if "split_label" in df else "defects"
    groups = df.groupby("study_id", sort=True)[split_column].max()
    try:
        train_ids, hold_ids = train_test_split(
            groups.index.to_numpy(), test_size=0.4, random_state=seed,
            stratify=groups.to_numpy(),
        )
        val_ids, test_ids = train_test_split(
            hold_ids, test_size=0.5, random_state=seed,
            stratify=groups.loc[hold_ids].to_numpy(),
        )
    except ValueError as exc:
        raise ValueError("Недостаточно исследований каждого класса для 60/20/20") from exc
    parts = [df[df.study_id.isin(ids)].reset_index(drop=True)
             for ids in (train_ids, val_ids, test_ids)]
    for name, part in zip(("train", "val", "test"), parts):
        if set(part.defects) != {0, 1}:
            raise ValueError(f"В {name} отсутствует один из классов")
        print(f"{name}: {len(part)}; классы {part.defects.value_counts().to_dict()}")
    return parts


def geometry_features(img):
    """Направление яркого центрального гребня при нескольких масштабах/ROI.

    Динамическое программирование штрафует скачки между строками; в отличие
    от argmax каждой строки не перескакивает свободно на рёбра. Не является
    сегментацией: ошибки гребня должен учитывать обучаемый классификатор.
    Абсолютные наклоны одинаково описывают отклонение вправо и влево.
    """
    arr = np.asarray(img, dtype=np.float64) / 255.0
    features = []
    h, w = arr.shape
    for top, bottom in ((0.12, 0.72), (0.22, 0.82)):
        y0, y1, x0, x1 = int(top*h), int(bottom*h), int(0.2*w), int(0.8*w)
        yy = np.arange(y0, y1, dtype=float)
        for sigma in (2.0, 5.0):
            smooth = gaussian_filter(arr, sigma=(2, sigma))
            background = gaussian_filter(arr, sigma=(2, 18))
            ridge = (smooth - background)[y0:y1, x0:x1]
            ridge /= max(float(np.std(ridge)), 1e-6)
            width = ridge.shape[1]
            xs = np.arange(width)
            # Мягкое предпочтение центра, не ограничение прямой осью.
            score = ridge[0] - 0.1 * ((xs-width/2)/(width/2))**2
            back = np.zeros(ridge.shape, dtype=np.int64)
            for row in range(1, len(ridge)):
                options = []
                for delta in range(-2, 3):
                    previous = xs + delta
                    valid = (previous >= 0) & (previous < width)
                    options.append(np.where(valid, score[np.clip(previous, 0, width-1)]
                                            - 0.15 * delta**2, -np.inf))
                options = np.asarray(options)
                best = options.argmax(axis=0)
                back[row] = np.clip(xs + best - 2, 0, width-1)
                score = ridge[row] + options[best, xs]
            path = np.zeros(len(ridge), dtype=np.int64)
            path[-1] = score.argmax()
            for row in range(len(ridge)-1, 0, -1):
                path[row-1] = back[row, path[row]]
            xx = median_filter(path.astype(float), size=7) + x0
            slope, intercept = np.polyfit(yy, xx, 1)
            residual = xx - (slope*yy + intercept)
            features.extend([
                abs(np.degrees(np.arctan(slope))), np.std(residual)/w,
                np.ptp(xx)/w, abs(xx[-1]-xx[0])/len(xx),
                np.mean((path == 0) | (path == width-1)),
                np.mean(ridge[np.arange(len(path)), path]),
            ])
            for section in np.array_split(np.arange(len(xx)), 3):
                local = np.polyfit(yy[section], xx[section], 1)[0]
                features.append(abs(np.degrees(np.arctan(local))))
            sampled = np.interp(np.linspace(0, len(xx)-1, 12), np.arange(len(xx)), xx)
            features.extend(np.abs(sampled - np.median(sampled))/w)
            features.extend(np.abs(np.diff(sampled))/w)
    result = np.asarray(features, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Нечисловые геометрические признаки")
    return result


def build_encoder(pretrained=False, kind="resnet18"):
    if kind == "xrv-densenet121":
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            try:
                import torchxrayvision as xrv
            except ImportError as exc:
                raise ImportError("Для --encoder xrv-densenet121: python -m pip install torchxrayvision") from exc
            medical = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.features.load_state_dict(medical.features.state_dict(), strict=True)
        # Инференс использует torchvision и сохранённые веса, без скачиваний XRV.
        encoder = nn.Sequential(model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)))
    elif kind == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        encoder = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d((2, 2)))
    else:
        raise ValueError(f"Неизвестный экстрактор: {kind}")
    encoder.encoder_kind = kind
    encoder.requires_grad_(False)
    return encoder.eval()


@torch.inference_mode()
def cnn_features(images, encoder, device):
    x = torch.stack([image_tensor(img) for img in images]).to(device)
    if getattr(encoder, "encoder_kind", "resnet18") == "xrv-densenet121":
        x = (x * 2 - 1) * 1024  # Нормализация TorchXRayVision: [-1024, 1024].
    else:
        x = x.repeat(1, 3, 1, 1)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        x = (x - mean) / std
    # Зеркалирование сохраняет |угол|. Вращения и растяжение не применяются.
    spatial = (encoder(x) + encoder(x.flip(-1)).flip(-1)) / 2
    symmetric = torch.cat((spatial.mean(-1), (spatial[..., 0]-spatial[..., 1]).abs()), dim=1)
    return symmetric.flatten(1).cpu().numpy().astype(np.float64)


def extract_features(images, encoder, device, batch_size):
    geom = np.stack([geometry_features(img) for img in images])
    cnn = np.concatenate([cnn_features(images[i:i+batch_size], encoder, device)
                          for i in range(0, len(images), batch_size)])
    return np.concatenate((geom, cnn), axis=1), geom.shape[1]


def cv_splits(frame, folds, repeats, seed):
    groups = frame.groupby("study_id", sort=True).defects.max()
    if groups.value_counts().min() < folds:
        raise ValueError(f"Недостаточно исследований каждого класса для {folds} фолдов")
    for repeat in range(repeats):
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed+repeat)
        for tr, va in splitter.split(groups.index, groups.values):
            train = np.flatnonzero(frame.study_id.isin(groups.index[tr]).to_numpy())
            val = np.flatnonzero(frame.study_id.isin(groups.index[va]).to_numpy())
            if set(frame.iloc[train].defects) != {0, 1} or set(frame.iloc[val].defects) != {0, 1}:
                raise ValueError("В одном из фолдов отсутствует класс")
            yield train, val


def make_estimator(config, n_geometry, n_features, n_samples, seed):
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if config["kind"] == "forest":
        return BalancedRandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=2,
            sampling_strategy="all", replacement=True, bootstrap=False,
            class_weight={0: 1.0, 1: config.get("positive_weight", 1.0)},
            random_state=seed, n_jobs=1,
        )
    blocks = [("geometry", StandardScaler(), slice(0, n_geometry))]
    if config["features"] == "combined":
        blocks.append(("cnn", Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=min(16, n_samples-2, n_features-n_geometry),
                         svd_solver="full", whiten=False)),
            ("pc_scale", StandardScaler()),
        ]), slice(n_geometry, n_features)))
    return Pipeline([
        ("features", ColumnTransformer(blocks)),
        # Ни PCA, ни sampler не видят validation/test. Не смешиваем пиксели SMOTE.
        ("balance", RandomOverSampler(random_state=seed)),
        # ROS балансирует частоты; дополнительный вес явно задаёт цену пропуска,
        # а не ещё одну автоматическую компенсацию отношения 89:10.
        ("classifier", LogisticRegression(C=config["C"], solver="lbfgs", max_iter=3000,
                                         class_weight={0: 1.0, 1: config.get("positive_weight", 1.0)})),
    ])


def export_estimator(estimator, config, n_geometry):
    """Только тензоры/числа: загрузка через weights_only=True, без pickle моделей."""
    state = {"kind": config["kind"], "features": config["features"]}
    def tensor(value):
        return torch.from_numpy(np.asarray(value).copy())
    if config["kind"] == "forest":
        state["trees"] = []
        for tree in estimator.estimators_:
            t = tree.tree_
            values = t.value[:, 0, :]
            state["trees"].append({
                "left": tensor(t.children_left), "right": tensor(t.children_right),
                "feature": tensor(t.feature), "threshold": tensor(t.threshold),
                "probability": tensor(values[:, 1]/values.sum(axis=1)),
            })
    else:
        blocks = estimator.named_steps["features"].named_transformers_
        geom = blocks["geometry"]
        state.update(geometry_mean=tensor(geom.mean_), geometry_scale=tensor(geom.scale_))
        if config["features"] == "combined":
            cnn = blocks["cnn"].named_steps
            state.update(cnn_mean=tensor(cnn["scale"].mean_), cnn_scale=tensor(cnn["scale"].scale_),
                         pca_mean=tensor(cnn["pca"].mean_), pca_components=tensor(cnn["pca"].components_),
                         pc_mean=tensor(cnn["pc_scale"].mean_), pc_scale=tensor(cnn["pc_scale"].scale_))
        clf = estimator.named_steps["classifier"]
        state.update(coef=tensor(clf.coef_[0]), intercept=float(clf.intercept_[0]))
    state["n_geometry"] = n_geometry
    return state


def predict_estimator(state, x):
    n_geometry = state["n_geometry"]
    def array(key):
        return state[key].numpy()
    if state["kind"] == "forest":
        # sklearn trees сравнивают признаки, приведённые к float32.
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
                nodes[active] = np.where(x[active, feature[current]] <= threshold[current],
                                         left[current], right[current])
            predictions.append(tree["probability"].numpy()[nodes])
        return np.mean(predictions, axis=0)
    geom = (x[:, :n_geometry]-array("geometry_mean"))/array("geometry_scale")
    if state["features"] == "combined":
        cnn = (x[:, n_geometry:]-array("cnn_mean"))/array("cnn_scale")
        cnn = (cnn-array("pca_mean")) @ array("pca_components").T
        cnn = (cnn-array("pc_mean"))/array("pc_scale")
        geom = np.concatenate((geom, cnn), axis=1)
    return expit(geom @ array("coef") + state["intercept"])


def predict_ensemble(states, x):
    return np.mean([predict_estimator(state, x) for state in states], axis=0)


class FeaturePredictor:
    def __init__(self, checkpoint, device):
        if checkpoint.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Несовместимая версия признаков")
        self.states, self.device = checkpoint["estimators"], device
        self.encoder = None
        if checkpoint["uses_cnn"]:
            self.encoder = build_encoder(kind=checkpoint.get("encoder_kind", "resnet18")).to(device)
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])

    def predict(self, image):
        x = geometry_features(image)[None, :]
        if self.encoder is not None:
            x = np.concatenate((x, cnn_features([image], self.encoder, self.device)), axis=1)
        return float(predict_ensemble(self.states, x)[0])


def select_threshold(labels, scores, beta=2.0, min_specificity=0.75):
    candidates = np.unique(np.r_[0.0, 0.5, scores, np.nextafter(scores.max(), np.inf)])
    candidates = [t for t in candidates if np.mean(scores[labels == 0] < t) >= min_specificity]
    return float(max(candidates, key=lambda t: (
        fbeta_score(labels, scores >= t, beta=beta, zero_division=0),
        recall_score(labels, scores >= t, zero_division=0),
        balanced_accuracy_score(labels, scores >= t), -abs(t - 0.5),
    )))


def metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=HERE / "datasets/dataset_for_spine_model.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_WEIGHTS,
                        help="Путь сохранения весов (по умолчанию spine_model/weights/best_model.pt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-weights", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--min-specificity", type=float, default=0.75)
    parser.add_argument("--encoder", choices=["resnet18", "xrv-densenet121"], default="resnet18")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.folds < 2 or min(args.batch_size, args.repeats) < 1:
        parser.error("folds >= 2, batch-size и repeats >= 1")
    if (not np.isfinite(args.beta) or args.beta <= 0 or not 0 <= args.min_specificity <= 1
            or any(not np.isfinite(w) or w < 1 for w in args.positive_weights)):
        parser.error("beta > 0, min-specificity в [0,1], positive-weights >= 1; значения конечные")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    frame = read_dataset(args.csv, args.data_root)
    corrected = frame[frame.defects != frame.split_label]
    if len(corrected):
        print("Применены подтверждённые исправления меток:")
        print(corrected[["rel_path", "split_label", "defects"]].to_string(index=False))
    train, val, test = split_dataset(frame, args.seed)
    # Сохраняем прежний test, но используем train+val для повторной CV.
    development = pd.concat((train, val), ignore_index=True)
    folds = list(cv_splits(development, args.folds, args.repeats, args.seed))
    encoder = build_encoder(pretrained=True, kind=args.encoder).to(args.device)
    print("Извлечение замороженных признаков development...", flush=True)
    images = [read_image(path) for path in development.path]
    x, n_geometry = extract_features(images, encoder, args.device, args.batch_size)
    y = development.defects.to_numpy()
    configs = [
        {"kind": "logistic", "features": features, "C": c}
        for features in ("geometry", "combined") for c in (0.1, 1.0)
    ] + [{"kind": "forest", "features": "geometry"}]
    configs = [dict(config, positive_weight=weight) for config in configs
               for weight in sorted(set(args.positive_weights))]
    best_rank, best = None, None
    summaries = []
    for config in configs:
        print(f"Кандидат: {config}", flush=True)
        oof_sum, counts = np.zeros(len(y)), np.zeros(len(y), dtype=int)
        states, fold_ap = [], []
        xx = x[:, :n_geometry] if config["features"] == "geometry" else x
        for number, (tr, va) in enumerate(folds):
            estimator = make_estimator(config, n_geometry, xx.shape[1], len(tr), args.seed+number)
            estimator.fit(xx[tr], y[tr])
            scores = estimator.predict_proba(xx[va])[:, 1]
            state = export_estimator(estimator, config, n_geometry)
            # Гарантия, что сохранённый формат воспроизводит sklearn/imblearn.
            np.testing.assert_allclose(predict_estimator(state, xx[va]), scores, atol=1e-7)
            states.append(state)
            oof_sum[va] += scores
            counts[va] += 1
            fold_ap.append(float(average_precision_score(y[va], scores)))
            print(f"  fold {number+1}/{len(folds)} AP={fold_ap[-1]:.4f}", flush=True)
        if not np.all(counts == args.repeats):
            raise RuntimeError("Неполное покрытие OOF")
        oof = oof_sum/counts
        candidate_threshold = select_threshold(y, oof, args.beta, args.min_specificity)
        objective = float(fbeta_score(y, oof >= candidate_threshold, beta=args.beta, zero_division=0))
        ap, auc = float(average_precision_score(y, oof)), float(roc_auc_score(y, oof))
        summary = {"config": config, "oof_AP": ap, "oof_AUC": auc,
                   "oof_Fbeta": objective, "threshold": candidate_threshold,
                   "recall": float(recall_score(y, oof >= candidate_threshold)),
                   "fold_AP_mean": float(np.mean(fold_ap)), "fold_AP_std": float(np.std(fold_ap))}
        summaries.append(summary)
        print("OOF (для выбора):", summary, flush=True)
        rank = (objective, ap, auc)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = (config, states, oof)
    config, states, oof = best
    threshold = select_threshold(y, oof, args.beta, args.min_specificity)
    # Порог выбран по каждому снимку только из моделей, не обученных на нём.
    oof_metrics = metrics(y, oof, threshold)
    print("Выбран:", config)
    print(f"Порог по OOF: {threshold:.6f}")
    print("OOF (подбор модели/порога, не независимая оценка):", oof_metrics)
    if oof_metrics["roc_auc"] <= 0.5 or oof_metrics["average_precision"] <= float(y.mean()):
        print("ВНИМАНИЕ: OOF не лучше случайного ранжирования. Проверьте метки и расширьте набор.")
    # Только теперь читаем test: он не влияет на выбор конфигурации и порога.
    test_images = [read_image(path) for path in test.path]
    uses_cnn = config["features"] == "combined"
    if uses_cnn:
        test_x, _ = extract_features(test_images, encoder, args.device, args.batch_size)
    else:
        test_x = np.stack([geometry_features(img) for img in test_images])
    test_scores = predict_ensemble(states, test_x)
    test_metrics = metrics(test.defects.to_numpy(), test_scores, threshold)
    checkpoint = {
        "architecture": "spine_feature_ensemble", "feature_version": FEATURE_VERSION,
        "preprocess": PREPROCESS, "class_names": ["tilt_le_5", "tilt_gt_5"],
        "estimators": states, "uses_cnn": uses_cnn, "threshold": threshold,
        "encoder_kind": args.encoder, "beta": args.beta, "min_specificity": args.min_specificity,
        "label_corrections": LABEL_CORRECTIONS,
        "encoder_state_dict": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                              if uses_cnn else {},
        "seed": args.seed, "folds": args.folds, "repeats": args.repeats,
        "selected_config": config, "candidate_metrics": summaries,
        "oof_selection_metrics": oof_metrics, "test_metrics": test_metrics,
        "oof_scores": torch.from_numpy(oof),
        "split_study_ids": {name: part.study_id.unique().tolist()
                            for name, part in (("development", development), ("test", test))},
        "cv_validation_study_ids": [development.iloc[va].study_id.unique().tolist() for _, va in folds],
        "versions": {"torch": str(torch.__version__), "sklearn": sklearn.__version__,
                     "imblearn": imblearn.__version__},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print("TEST (прежняя выборка, только spine_model):", test_metrics)
    print(f"Сохранено: {output}")
    print(f"Test: {int(test.defects.sum())} отклонений; нужна внешняя проверка.")


if __name__ == "__main__":
    main()
