"""Оценка сохранённых моделей по ТЗ.pdf (п. 8.4), без обучения.

Запуск: python main.py
Параметры: --data-root PATH --output PATH --bootstrap 1000 --seed 42
Только аудит CSV/путей, без загрузки моделей: python main.py --validate-only

Все пути по умолчанию вычисляются относительно этого файла. CSV не изменяется.
1 всегда означает нарушение, в том числе в столбцах «корректная ...».
В отчёте только метрики: по нарушениям/отделам, общая бинарная оценка и
определение отдела. Используется автоматическая маршрутизация BodyPartModel.
Общая эталонная метка берётся ТОЛЬКО из столбца 16.
ROI оценивается вероятностной головой HipQualityModel; отдельный CV PositionModel
не включается в этот ансамбль. Это не оценка полного веб-сервиса с CV-проверкой.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import math
import os
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parent.parent
LABELS = (
    "Позвоночник: нарушение укладки",
    "Позвоночник: наклон оси > 5 градусов",
    "Позвоночник: артефакты / посторонние предметы",
    "Правое бедро: позиционирование / ротация",
    "Правое бедро: некорректная ROI",
    "Левое бедро: позиционирование / ротация",
    "Левое бедро: некорректная ROI",
    "Общая: есть хотя бы одно нарушение (последний столбец)",
)
EXPECTED_HEADERS = (
    "study_id", "series_id", "file_name", "rel_path", "pixel_hash",
    "body_spine", "body_hip_right", "body_hip_left",
    "корректная укладка(позвоночник)",
    "правильно выравнена ось позвоночника (до 5)",
    "наличие посторонних предметов, выраженных артефактов или наложений",
    "позиционирование/ротация ППОБ", "корректности области интересов ППОБ",
    "позиционирование/ротация ЛПОБ", "корректности области интересов ЛПОБ",
    "Есть ли ошибка",
)


def read_csv(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("CSV должен иметь кодировку UTF-8 или Windows-1251")
    reader = csv.reader(io.StringIO(text), delimiter=";", strict=True)
    records = []
    for cells in reader:
        records.append((reader.line_num, cells))
    return records, encoding, hashlib.sha256(raw).hexdigest()


def find_dicom(rel_path, roots):
    relative = Path(rel_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("rel_path должен быть относительным, без '..'")
    # В CSV сохранён префикс НД_для_обучения/Исследования; локально архив
    # распакован в Data/Исследования. Сохраняем весь путь от study_id, не basename.
    variants = [relative]
    if "Исследования" in relative.parts:
        index = relative.parts.index("Исследования")
        variants.extend((Path(*relative.parts[index:]), Path(*relative.parts[index + 1:])))
    for root in roots:
        matches = {candidate.resolve() for variant in variants
                   if (candidate := root / variant).is_file()}
        if len(matches) > 1:
            raise ValueError(f"Неоднозначный путь: {sorted(map(str, matches))}")
        if matches:
            return matches.pop()
    raise FileNotFoundError(rel_path)


def audit(dataset, roots, report):
    records, encoding, digest = read_csv(dataset)
    report(f"CSV: {dataset}\nКодировка: {encoding}\nSHA256: {digest}")
    if not records or tuple(c.strip() for c in records[0][1]) != EXPECTED_HEADERS:
        raise ValueError("Неожиданный состав или порядок 16 столбцов CSV")
    rows, errors, warnings = [], [], []
    seen = {key: {} for key in ("path", "hash", "identity")}
    for line, original in records[1:]:
        cells = [c.strip() for c in original]
        prefix = f"Строка {line}"
        if len(cells) != 16:
            errors.append(f"{prefix}: {len(cells)} столбцов вместо 16")
            continue
        if original != cells:
            warnings.append(f"{prefix}: внешние пробелы удалены только в памяти")
        if any(not c for c in cells[:5]):
            errors.append(f"{prefix}: пустые идентификаторы/путь/хеш")
        if any(c not in {"0", "1"} for c in cells[5:]):
            errors.append(f"{prefix}: метки должны быть ровно 0 или 1: {cells[5:]}")
            continue
        body = tuple(map(int, cells[5:8]))
        truth = tuple(map(int, cells[8:]))
        if not any(body) or (body[0] and any(body[1:])):
            errors.append(f"{prefix}: несовместимая/отсутствующая анатомия {body}")
        if body[1] and body[2]:
            warnings.append(f"{prefix}: одновременно правое и левое бедро; проверить вручную. "
                            "Из метрик определения отдела строка исключается; BodyPartModel выбирает одну сторону.")
        applicable = (body[0],) * 3 + (body[1],) * 2 + (body[2],) * 2
        for j, active in enumerate(applicable):
            if truth[j] and not active:
                errors.append(f"{prefix}: нарушение вне отмеченной области: {LABELS[j]}")
        if truth[-1] != int(any(truth[:-1])):
            errors.append(f"{prefix}: последний столбец {truth[-1]} != OR семи меток {truth[:-1]}")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", cells[4]):
            warnings.append(f"{prefix}: pixel_hash не похож на MD5 из 32 hex-символов; "
                            "алгоритм хеширования CSV неизвестен")
        relative = cells[3].replace("\\", "/")
        if Path(relative).name != cells[2] or cells[0] not in Path(relative).parts \
                or cells[1] not in Path(relative).parts:
            errors.append(f"{prefix}: rel_path не согласован с study_id/series_id/file_name")
        try:
            path = find_dicom(relative, roots)
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"{prefix}: DICOM не найден или неоднозначен: {exc}")
            path = None
        values = {"path": str(path) if path else relative, "hash": cells[4].lower(),
                  "identity": tuple(cells[:3])}
        for key, value in values.items():
            if value in seen[key]:
                previous, labels = seen[key][value]
                conflict = "; ПРОТИВОРЕЧИВЫЕ МЕТКИ" if labels != cells[5:] else ""
                errors.append(f"{prefix}: дубликат {key} строки {previous}{conflict}")
            else:
                seen[key][value] = (line, cells[5:])
        rows.append(dict(line=line, study=cells[0], path=path, body=body,
                         truth=truth, applicable=applicable))
    report(f"\nАудит CSV: строк={len(records)-1}, ошибок={len(errors)}, предупреждений={len(warnings)}")
    for issue in errors + warnings:
        report(issue)
    if not rows:
        errors.append("Нет строк для оценки")
    for j, name in enumerate(LABELS):
        eligible = [r for r in rows if j == 7 or r["applicable"][j]]
        report(f"{name}: N={len(eligible)}, положительных={sum(r['truth'][j] for r in eligible)}")
    if errors:
        raise ValueError("Аудит не пройден. Исправьте ошибки CSV/путей; метрики не рассчитаны.")
    # Совпадение с исходным набором обучения не доказывает попадание именно в train,
    # поскольку разбиение могло выполняться позднее внутри train.py.
    studies = {r["study"] for r in rows}
    for source in sorted((PROJECT / "models").glob("*/datasets/*.csv")):
        try:
            raw = source.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1251")
            delimiter = ";" if ";" in text.splitlines()[0] else ","
            training = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            overlap = studies & {r.get("study_id", "").strip() for r in training}
            if overlap:
                report(f"ПРЕДУПРЕЖДЕНИЕ: {len(overlap)} study_id встречаются в {source}. "
                       "Независимость тестовой выборки не подтверждена.")
        except Exception as exc:
            report(f"Проверка пересечения с {source} недоступна: {exc}")
    report("Аудит не проверяет клиническую правильность ручных меток и содержимое pixel_hash.")
    return rows


def check_dicom_metadata(row, report):
    """CSV study_id идентифицирует папку, а не обязательно внутренний DICOM UID.

    После обезличивания UID может отличаться. Связь с разметкой проверяется
    в audit по полному rel_path и его study_id/series_id/file_name. Для группировки
    и bootstrap сохраняем исходный study_id CSV. Ошибки чтения не подавляются.
    """
    import pydicom

    ds = pydicom.dcmread(str(row["path"]), stop_before_pixels=True)
    uid = str(getattr(ds, "StudyInstanceUID", "")).strip()
    if uid != row["study"]:
        report(f"Строка {row['line']}: ПРЕДУПРЕЖДЕНИЕ: study_id папки/CSV="
               f"{row['study']}; StudyInstanceUID DICOM={uid or '<отсутствует>'}. "
               "Сопоставление выполнено по проверенному rel_path; инференс продолжается.")


def load_models(report):
    path = PROJECT / "dxa_web_app" / "inference_models.py"
    spec = importlib.util.spec_from_file_location("metrics_inference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Невозможно импортировать {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    definitions = {
        "body": (module.BodyPartModel, "body_part_model/weights/best_model.pt"),
        "axis": (module.SpineModel, "spine_model/weights/best_model.pt"),
        "placement": (module.SpinePositionModel, "spine_position_model/final_spine_model.pth"),
        "artifact": (module.ArtifactModel, "artifact_model/weights/best_artifact_model.pth"),
        "hip": (module.HipQualityModel, "vertel_model/weights/best_hip_quality.pt"),
    }
    models = {}
    for name, (cls, relative) in definitions.items():
        try:
            models[name] = cls(str(PROJECT / "models" / relative))
            model = models[name]
            report(f"Модель {name}: {relative}; пороги="
                   f"{getattr(model, 'thresholds', getattr(model, 'threshold', 'argmax'))}")
        except Exception as exc:
            report(f"ОШИБКА загрузки {name}: {type(exc).__name__}: {exc}")
    return module, models


def predict_region(row, models):
    import numpy as np

    result = models["body"].predict(str(row["path"]))
    if result.get("status") != "success":
        raise RuntimeError(f"Определение отдела: {result.get('error', 'ошибка модели')}")
    names = ("spine", "hip_right", "hip_left")
    probabilities = np.asarray(result["probabilities"], dtype=float)
    if probabilities.shape != (3,) or not np.isfinite(probabilities).all() \
            or not ((probabilities >= 0) & (probabilities <= 1)).all() \
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-5):
        raise ValueError("Некорректные вероятности определения отдела")
    label = names.index(result["class_name"])
    if label != int(probabilities.argmax()):
        raise ValueError("Класс отдела не согласован с вероятностями")
    return label, probabilities


def aggregate_violation(predictions, scores):
    """Положительная проверка достаточна, даже если другая завершилась сбоем."""
    import numpy as np

    if np.any(predictions == 1):
        label = 1.0
    elif np.isfinite(predictions).all():
        label = 0.0
    else:
        label = np.nan
    # Неполный набор вероятностей нельзя выдавать за полный score ансамбля.
    score = float(np.max(scores)) if np.isfinite(scores).all() else np.nan
    return label, score


def infer(row, module, models, region_id, report):
    """Возвращает метки/score; NaN означает сбой, а не отрицательный ответ."""
    import numpy as np
    import torch

    scores = np.zeros(8, dtype=float)
    predictions = np.zeros(8, dtype=float)
    body = tuple(int(region_id == index) for index in range(3))

    def predict(name, **kwargs):
        if name not in models:
            raise RuntimeError(f"Модель {name} не загружена")
        result = models[name].predict(str(row["path"]), **kwargs)
        if result.get("status") != "success":
            raise RuntimeError(f"{name}: {result.get('error', 'неизвестная ошибка')}")
        return result

    def binary(name, index):
        try:
            result = predict(name)
            probability = float(result["probability"])
            label = result["class_id"]
            if not math.isfinite(probability) or not 0 <= probability <= 1 or label not in (0, 1):
                raise ValueError("Некорректная вероятность/метка")
            scores[index], predictions[index] = probability, label
        except Exception as exc:
            scores[index] = predictions[index] = np.nan
            report(f"Строка {row['line']}, {name}: {exc}")

    if body[0]:
        binary("placement", 0)
        binary("axis", 1)
        # В данной CSV артефакты относятся к позвоночнику. Не добавляем к
        # общей оценке бедра проверку, для которой там нет отдельной разметки.
        binary("artifact", 2)
    for active, side, index in ((body[1], "right", 3), (body[2], "left", 5)):
        if not active:
            continue
        try:
            model = models["hip"]
            # train.py: side_id = (side == 'right').astype(int): right=1, left=0.
            # Не используем ошибочное отображение right=0 из HipQualityModel.predict.
            arr = module.load_dicom_minmax(row["path"])
            img = module.to_square_gray(arr, model.img_size)
            x = torch.from_numpy(np.array(img, copy=True)).float()[None, None] / 255.0
            x = ((x - 0.5) / 0.25).to(model.device)
            side_tensor = torch.tensor([int(side == "right")], device=model.device)
            with torch.inference_mode():
                probability = torch.sigmoid(model.model(x, side_tensor)).cpu().numpy()[0]
            thresholds = np.asarray(model.thresholds, dtype=float)
            if probability.shape != (2,) or thresholds.shape != (2,) \
                    or not np.isfinite(probability).all() or not np.isfinite(thresholds).all() \
                    or not ((thresholds >= 0) & (thresholds <= 1)).all():
                raise ValueError("Некорректные выходы/пороги двух голов HipQualityModel")
            scores[index:index+2] = probability
            predictions[index:index+2] = probability >= thresholds
        except Exception as exc:
            scores[index:index+2] = predictions[index:index+2] = np.nan
            report(f"Строка {row['line']}, hip/{side}: {exc}")
    predictions[7], scores[7] = aggregate_violation(predictions[:7], scores[:7])
    return predictions, scores


def binary_metrics(y, pred, scores):
    import numpy as np

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    ratio = lambda a, b: a / b if b else np.nan
    sensitivity, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    result = dict(sensitivity=sensitivity, specificity=specificity,
                  balanced_accuracy=(sensitivity+specificity)/2,
                  precision=ratio(tp, tp+fp), F1=ratio(2*tp, 2*tp+fp+fn),
                  accuracy=ratio(tp+tn, len(y)), ROC_AUC=np.nan, PR_AUC=np.nan)
    score_mask = np.isfinite(scores)
    y, scores = y[score_mask], scores[score_mask]
    if len(np.unique(y)) == 2:
        # Считаем точки кривых по уникальным порогам: равные scores включаются
        # одновременно, поэтому порядок примеров с одинаковым score не влияет на AUC.
        order = np.argsort(-scores, kind="stable")
        sorted_y, sorted_scores = y[order], scores[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
        true_positives = np.cumsum(sorted_y)[ends]
        false_positives = ends + 1 - true_positives
        recall = np.r_[0.0, true_positives / y.sum()]
        fpr = np.r_[0.0, false_positives / (len(y) - y.sum())]
        precision = np.r_[1.0, true_positives / (ends + 1)]
        result["ROC_AUC"] = float(np.sum(np.diff(fpr) * (recall[:-1] + recall[1:]) / 2))
        result["PR_AUC"] = float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))
    return result


def summarize(rows, predictions, scores, bootstrap, seed, report):
    import numpy as np

    truth = np.asarray([r["truth"] for r in rows])
    applicable = np.asarray([(*r["applicable"], 1) for r in rows], dtype=bool)
    groups = np.asarray([r["study"] for r in rows])
    summarize_binary_tasks(
        truth, predictions, scores, applicable, groups, LABELS, bootstrap, seed, report,
        macro_count=7, macro_label="Macro-F1 (7 типов, равные веса, только применимые области)",
    )


def summarize_binary_tasks(truth, predictions, scores, applicable, groups, names,
                           bootstrap, seed, report, macro_count, macro_label):
    """Общий формат метрик нарушений и определения отдела (один против остальных)."""
    import numpy as np

    tasks = [(name, j, applicable[:, j]) for j, name in enumerate(names)]
    # Единые кластерные bootstrap-веса сохраняют зависимость снимков и меток
    # одного исследования, включая bilateral. Не bootstrap отдельных изображений.
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = []
    grouped = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    for _ in range(bootstrap):
        samples.append(np.concatenate([grouped[i] for i in rng.integers(0, len(unique), len(unique))]))

    def interval(values):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(unique) < 2 or len(finite) < max(20, math.ceil(bootstrap * 0.5)):
            return f"95% ДИ: N/A (валидных повторов {len(finite)}/{bootstrap})"
        low, high = np.percentile(finite, [2.5, 97.5])
        return f"95% ДИ [{low:.4f}; {high:.4f}], повторов {len(finite)}/{bootstrap}"

    f1_values, f1_bootstrap = [], []
    for task_index, (name, j, eligible) in enumerate(tasks):
        mask = eligible & np.isfinite(predictions[:, j])
        y, p, s = truth[mask, j], predictions[mask, j], scores[mask, j]
        report(f"\n{name}\nN={len(y)}/{int(eligible.sum())}; "
               f"сбоев={int(eligible.sum()-mask.sum())}; positive={int(y.sum())}; "
               f"negative={int(len(y)-y.sum())}; N_AUC={int(np.isfinite(s).sum())}")
        counts = [int(((y == a) & (p == b)).sum()) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        report(f"TN={counts[0]}, FP={counts[1]}, FN={counts[2]}, TP={counts[3]}")
        point = binary_metrics(y, p, s)
        draws = defaultdict(list)
        for indices in samples:
            take = indices[mask[indices]]
            metrics = binary_metrics(truth[take, j], predictions[take, j], scores[take, j])
            for metric, value in metrics.items():
                draws[metric].append(value)
        for metric, value in point.items():
            formatted = f"{value:.4f}" if np.isfinite(value) else "N/A"
            ci = interval(draws[metric]) if np.isfinite(value) else "95% ДИ: N/A"
            report(f"  {metric}: {formatted}; {ci}")
        if task_index < macro_count:
            f1_values.append(point["F1"])
            f1_bootstrap.append(draws["F1"])
    # Не nanmean: отсутствие целого класса/модели не должно улучшать macro-F1.
    macro = float(np.mean(f1_values))
    macro_draws = np.mean(np.asarray(f1_bootstrap), axis=0) if bootstrap else []
    report(f"\n{macro_label}: "
           f"{macro:.4f}; {interval(macro_draws)}")
    if not np.isfinite(macro):
        report("Macro-F1 недоступна: F1 хотя бы одного класса не определена.")


def multiclass_region_metrics(targets, predictions, probabilities):
    """Метрики трёх фиксированных классов; AUC усредняется по схеме OvR."""
    import numpy as np

    names = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
             "F1_macro", "F1_weighted", "ROC_AUC_macro_OvR", "PR_AUC_macro_OvR")
    if not len(targets):
        return dict.fromkeys(names, np.nan)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (targets.astype(int), predictions.astype(int)), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = matrix.diagonal()
    # Как zero_division=0: отсутствие положительных прогнозов не должно
    # исключать трудный класс из macro precision / F1.
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(tp, support, out=np.full(3, np.nan), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    aucs = [binary_metrics((targets == j).astype(int), (predictions == j).astype(int),
                           probabilities[:, j]) for j in range(3)]
    return dict(zip(names, (
        float(tp.sum() / len(targets)), float(recall.mean()), float(precision.mean()),
        float(recall.mean()), float(f1.mean()), float(np.sum(f1 * support) / len(targets)),
        float(np.mean([a["ROC_AUC"] for a in aucs])),
        float(np.mean([a["PR_AUC"] for a in aucs])),
    )))


def summarize_regions(rows, predictions, probabilities, bootstrap, seed, report):
    """Один мультиклассовый блок; неоднозначные эталоны исключаются."""
    import numpy as np

    truth = np.asarray([r["body"] for r in rows])
    eligible = truth.sum(axis=1) == 1
    valid = eligible & np.isin(predictions, [0, 1, 2])
    targets = truth.argmax(axis=1)
    report(f"\nОпределение отдела (мультиклассовая модель: позвоночник, правое бедро, левое бедро)"
           f"\nN={int(valid.sum())}/{int(eligible.sum())}; сбоев={int((eligible & ~valid).sum())}; "
           f"неоднозначных меток={int((~eligible).sum())}; "
           f"N_AUC={int((valid & np.isfinite(probabilities).all(axis=1)).sum())}")

    def metrics(indices):
        return multiclass_region_metrics(targets[indices], predictions[indices], probabilities[indices])

    point = metrics(np.flatnonzero(valid))
    groups = np.asarray([r["study"] for r in rows])
    unique = np.unique(groups[valid])
    grouped = [np.flatnonzero(valid & (groups == study)) for study in unique]
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    if len(unique) >= 2:
        for _ in range(bootstrap):
            indices = np.concatenate([grouped[j] for j in rng.integers(0, len(unique), len(unique))])
            for key, value in metrics(indices).items():
                draws[key].append(value)
    for key, value in point.items():
        formatted = f"{value:.4f}" if np.isfinite(value) else "N/A"
        finite = np.asarray(draws[key])
        finite = finite[np.isfinite(finite)]
        ci = "95% ДИ: N/A"
        if np.isfinite(value) and len(finite) >= max(20, math.ceil(bootstrap * 0.5)):
            low, high = np.percentile(finite, [2.5, 97.5])
            ci = f"95% ДИ [{low:.4f}; {high:.4f}], повторов {len(finite)}/{bootstrap}"
        report(f"  {key}: {formatted}; {ci}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=HERE / "datasets" / "dataset_for_all_models.csv")
    parser.add_argument("--data-root", type=Path, help="Корень DICOM; приоритет над стандартными путями")
    parser.add_argument("--output", type=Path, default=HERE / "metrics_report.txt")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap должен быть >= 0")
    if args.output.resolve() == args.dataset.resolve():
        parser.error("Отчёт не должен перезаписывать CSV")

    def diagnostic(message):
        # Служебные записи и ошибки не попадают в файл с метриками.
        print(message, file=sys.stderr, flush=True)

    try:
        roots = [args.data_root.resolve()] if args.data_root else []
        roots += [args.dataset.resolve().parent, HERE / "dataset", REPO / "Data", REPO, PROJECT]
        rows = audit(args.dataset.resolve(), roots, diagnostic)
        if args.validate_only:
            print("Проверка завершена; модели не запускались.")
            return 0
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        module, models = load_models(diagnostic)
        predictions, scores, region_predictions, region_scores = [], [], [], []
        for index, row in enumerate(rows, 1):
            region, region_score = np.nan, np.full(3, np.nan)
            pred, score = np.full(8, np.nan), np.full(8, np.nan)
            try:
                check_dicom_metadata(row, diagnostic)
                region, region_score = predict_region(row, models)
                pred, score = infer(row, module, models, region, diagnostic)
            except Exception as exc:
                diagnostic(f"Строка {row['line']}: {type(exc).__name__}: {exc}")
            predictions.append(pred)
            scores.append(score)
            region_predictions.append(region)
            region_scores.append(region_score)
            if index % 10 == 0 or index == len(rows):
                print(f"Обработано {index}/{len(rows)}", flush=True)
        predictions, scores = np.asarray(predictions), np.asarray(scores)
        region_predictions, region_scores = np.asarray(region_predictions), np.asarray(region_scores)
        print("Расчёт метрик и доверительных интервалов...", flush=True)
        # Сначала формируем полный отчёт в памяти, чтобы сбой вычислений не
        # оставлял на месте предыдущего отчёта частичный файл.
        lines = []
        summarize(rows, predictions, scores, args.bootstrap, args.seed, lines.append)
        summarize_regions(rows, region_predictions, region_scores, args.bootstrap, args.seed, lines.append)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines).lstrip() + "\n", encoding="utf-8")
        print(f"Отчёт: {args.output.resolve()}", flush=True)
        if not np.isfinite(predictions[:, 7]).any():
            diagnostic("Общая оценка недоступна для всех снимков. См. ошибки в консоли.")
            return 1
        return 0
    except Exception:
        diagnostic(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
