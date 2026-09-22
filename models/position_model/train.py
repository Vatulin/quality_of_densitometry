"""body_part_model определяет часть тела/сторону; отступы измеряются классическим CV.

ТЗ.pdf, п. 2.3, рис. 6: >=30 мм над большим вертелом, >=30 мм под
седалищной костью, >=20 мм с наружной стороны бедра. Не использовать
bounding box всей кости: диафиз закономерно доходит до нижнего края кадра.
Размер исходного пикселя: Y=1.05 мм, X=0.6 мм. Масштаб не изменяется.

Алгоритм: нормализация яркости, Gaussian blur, порог по Otsu (три уровня),
морфологическое замыкание, удаление мелких компонент; поиск диафиза в нижней
части кадра, выбор наружного края по стороне из body_part_model,
верхняя огибающая большого вертела и нижний отдельный медиальный
контур седалищной кости. Согласованность трёх сегментаций проверяется в мм.
Это эвристические анатомические ориентиры, не гарантированная сегментация.
Неоднозначное заключение, неподдерживаемые изображения и пограничные измерения
дают quality_class=None / Failure, а не ложное заключение об отсутствии дефекта.
Проверяется только поле обзора, не ротация и не вся категория position_defects.

Для исходных экспортов GE в этом датасете hip_left соответствует правому
краю изображения, hip_right — левому. Произвольные зеркальные экспорты
требуют проверки этой конвенции. side влияет на поиск, а не только на отчёт.
analyze_image(gray, side='auto') сохраняет чистый CV-режим для сравнения;
analyze и evaluate всегда сначала вызывают body_part_model.

Зависимости: numpy, opencv-python, pydicom, torch, torchvision, pillow.
python -B train.py --evaluate
CSV/JSON печатаются в stdout; файлы и обученные модели не создаются.
При предсказании CSV, имена файлов и метки не используются. Текущий датасет
использовался для разработки эвристики: метрики на нём не независимые.
Оценка по position_defects ориентировочная: нет разметки самих ориентиров.
Алгоритм пока не подтверждён как детектор position_defects.
Нужна отдельная разметка отступов/ориентиров. Диапазон по трём порогам —
проверка чувствительности сегментации, не статистический доверительный интервал.
"""

import argparse
import csv
import io
import json
import time
import sys
import importlib.util
from functools import lru_cache

sys.dont_write_bytecode = True
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, url2pathname

import cv2
import numpy as np
import pydicom

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[3] / "Data"
SPACING_X = 0.6
SPACING_Y = 1.05
REQUIRED_MM = {"top": 30.0, "bottom": 30.0, "lateral": 20.0}
MAX_BYTES = 64 * 1024 * 1024
CONTOUR_FACTORS = (.08, .12, .16)


BODY_MODEL_DIR = HERE.parent / "body_part_model"
DEFAULT_MODEL = BODY_MODEL_DIR / "weights" / "best_model.pt"


@lru_cache(maxsize=1)
def body_model_module():
    """Единый источник архитектуры, порядка классов и размера входа."""
    path = BODY_MODEL_DIR / "test.py"
    spec = importlib.util.spec_from_file_location("_position_body_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Невозможно загрузить модуль классификатора: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _load_body_model(path, device, modified_ns, size_bytes):
    """Одни веса на пакет; изменение файла автоматически меняет ключ кеша."""
    import torch
    model = body_model_module().build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def classify_body_part(data, model_path=DEFAULT_MODEL, device="cpu"):
    """Та же архитектура и preprocessing, что в body_part_model/test.py.

    Min/max исходного pixel_array, чёрный квадрат, bilinear до IMG_SIZE.
    Предобработка CV здесь не используется: это изменило бы вход модели.
    Веса локальные, скачивание/обучение отсутствуют.
    """
    import cv2
    import numpy as np
    import pydicom
    import torch
    from PIL import Image

    classifier = body_model_module()
    class_names = tuple(classifier.CLASS_NAMES)
    if set(class_names) != {"spine", "hip_right", "hip_left"} or len(class_names) != 3:
        raise ValueError(f"Неожиданные классы body_part_model: {class_names}")

    raster = data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8", b"BM", b"II*\x00", b"MM\x00*"))
    if raster:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        ds = pydicom.dcmread(io.BytesIO(data))
        if int(ds.get("NumberOfFrames", 1)) != 1 or int(ds.get("SamplesPerPixel", 1)) != 1:
            raise ValueError("Нужен однокадровый монохромный DICOM")
        arr = ds.pixel_array
    if arr is None or arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError("Нужен одноканальный снимок с числовыми пикселями")
    arr = arr.astype(np.float32)
    arr -= arr.min()
    if arr.max() <= 0:
        raise ValueError("Пустое изображение")
    arr /= arr.max()
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    square = classifier.pad_to_square(img)
    img = square.resize((classifier.IMG_SIZE, classifier.IMG_SIZE), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(img, copy=True)).float()[None, None] / 255.0
    model_path = Path(model_path).resolve()
    stat = model_path.stat()
    model = _load_body_model(str(model_path), device, stat.st_mtime_ns, stat.st_size)
    with torch.inference_mode():
        probs = torch.softmax(model(x.to(device)), dim=1)[0].cpu().numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Модель вернула нечисловые вероятности")
    return {"class": class_names[int(probs.argmax())],
            "probabilities": dict(zip(class_names, map(float, probs))),
            "model_path": str(model_path)}



def read_source(source):
    """Прочитать источник один раз, без промежуточных файлов."""
    source = str(source)
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        with urlopen(source, timeout=30) as response:
            data = response.read(MAX_BYTES + 1)
    else:
        if parsed.scheme == "file":
            source = url2pathname(parsed.path)
            if parsed.netloc:
                source = "//" + parsed.netloc + source
        path = Path(source)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Изображение превышает 64 MiB")
        data = path.read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("Изображение превышает 64 MiB")
    return data


def read_image(source, data=None):
    """Исходная немасштабированная матрица сканера, в RAM.

    Скриншоты, поворот, multi-frame и RGB DICOM не поддерживаются.
    data позволяет повторно использовать уже загруженный снимок.
    """
    if data is None:
        data = read_source(source)
    meta = {"study_uid": "", "image_uid": "", "laterality": ""}
    raster = data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8", b"BM", b"II*\x00", b"MM\x00*"))
    if raster:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None or arr.ndim != 2:
            raise ValueError("Нужна исходная одноканальная матрица, не цветной снимок")
    else:
        ds = pydicom.dcmread(io.BytesIO(data))
        if int(ds.get("NumberOfFrames", 1)) != 1 or int(ds.get("SamplesPerPixel", 1)) != 1:
            raise ValueError("Поддерживается только однокадровый монохромный DICOM")
        photo = str(ds.get("PhotometricInterpretation", ""))
        if photo not in ("MONOCHROME1", "MONOCHROME2"):
            raise ValueError("Неподдерживаемая фотометрическая интерпретация")
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            raise ValueError("Ожидалась двумерная матрица")
        arr = arr * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
        if photo == "MONOCHROME1":
            arr = arr.max() + arr.min() - arr
        spacing = ds.get("PixelSpacing")
        if spacing is not None and not np.allclose(np.asarray(spacing, float), [SPACING_Y, SPACING_X], rtol=0.02):
            raise ValueError("PixelSpacing отличается от заданных 1.05/0.6 мм; проверьте масштаб")
        meta.update(study_uid=str(ds.get("StudyInstanceUID", "")),
                    image_uid=str(ds.get("SOPInstanceUID", "")),
                    laterality=str(ds.get("ImageLaterality", ds.get("Laterality", ""))))
    if min(arr.shape) < 64 or max(arr.shape) > 4096 or not np.isfinite(arr).all():
        raise ValueError("Неподдерживаемый размер или нечисловые пиксели")
    lo, hi = np.percentile(arr, [1, 99.5])
    if hi <= lo:
        raise ValueError("Пустое изображение или недостаточный контраст")
    gray = np.clip((arr.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return gray, meta


def _runs(row):
    edges = np.diff(np.r_[False, row, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _bone_mask(smooth, threshold, factor):
    """Низкий порог сохраняет слабоконтрастный край; высокий задаёт диафиз."""
    h, w = smooth.shape
    mask = (smooth > max(3, threshold * factor)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= max(25, h * w * 0.002))
    return np.isin(labels, keep[keep != 0])


def _landmarks(gray, factor, side="auto"):
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = _bone_mask(smooth, threshold, factor)
    # Положение диафиза определяем отдельно по яркой кости: слабый сигнал
    # фона не должен менять сторону и ось при поиске наружного контура.
    core = _bone_mask(smooth, threshold, .55)
    # Диафиз — один достаточно широкий устойчивый интервал внизу кадра.
    shafts = []
    for y in range(int(h * .78), int(h * .96)):
        runs = [(a, b) for a, b in _runs(core[y]) if 12 <= (b-a)*SPACING_X <= 65]
        if len(runs) == 1:
            shafts.append(runs[0])
        elif runs and side != "auto":
            # Когда внизу ещё виден таз, диафиз ищем с наружной стороны,
            # известной по body_part_model, а не отбрасываем всю строку.
            shafts.append((max if side == "left" else min)(runs, key=lambda r: r[0]+r[1]))
    if len(shafts) < h * .08:
        raise ValueError("Диафиз не выделен однозначно: возможен обрезанный/повёрнутый кадр")
    shaft_left, shaft_right = np.median(shafts, axis=0)
    # right — исключённая граница интервала; ось проходит по центрам пикселей.
    shaft_x = (shaft_left + shaft_right - 1) / 2
    yy, xx = np.nonzero(core[:int(h * .23)])
    if side == "auto":
        if len(xx) < 30:
            raise ValueError("Не найден верхний контур таза")
        pelvis_x = float(np.median(xx))
        if abs(pelvis_x - shaft_x) * SPACING_X < 8:
            raise ValueError("Нельзя определить наружную сторону бедра")
        lateral_right = pelvis_x < shaft_x
    else:
        lateral_right = side == "left"
    # Приводим к виду: таз слева, диафиз справа. Размер пикселя не меняется.
    if not lateral_right:
        mask = mask[:, ::-1]
        core = core[:, ::-1]
        shaft_left, shaft_right = w - shaft_right, w - shaft_left
        shaft_x = w - 1 - shaft_x
    # Вершина большого вертела: верхний костный контур наружнее оси диафиза.
    outer = mask.copy()
    outer[:, :int(np.ceil(shaft_x))] = False
    # Слабые фоновые мостики могут соединить таз и вертел. Разделение
    # на анатомические ветви делаем по яркому ядру, уточняем край по mask.
    core_outer = core.copy()
    core_outer[:, :int(np.ceil(shaft_x))] = False
    count = core_outer.sum(axis=1)
    # Верхний таз иногда пересекает ось диафиза. Выбираем протяжённую
    # нижнюю ветвь профиля после межкостного промежутка, а не первый пиксель таза.
    candidates = [int(a) for a, b in _runs(count >= 5)
                  if a < h * .65 and b > h * .70 and b-a >= h * .20]
    if not candidates:
        raise ValueError("Не найден большой вертел")
    top = candidates[0]
    if top < 2:
        raise ValueError("Верхний ориентир касается края: измерение недостоверно")
    # Ограниченное расширение от уверенного ядра, не захватывающее таз.
    for y in range(top-1, max(-1, top-11), -1):
        if outer[y].sum() < 5:
            break
        top = y
    # Седалищная кость отделена от диафиза тёмным промежутком ниже шейки.
    medial_mask = np.zeros_like(mask, dtype=np.uint8)
    for y in range(top + 5, h):
        runs = _runs(mask[y])
        femur = [(a, b) for a, b in runs if a <= shaft_x < b]
        if len(femur) != 1:
            continue
        femur_left = femur[0][0]
        # Острый нижний конец может занимать 1–4 пикселя в строке.
        # Отбрасывание таких строк завышало нижний отступ. Шум отсечён
        # связными компонентами и требованием последовательных строк ниже.
        medial = [(a, b) for a, b in runs if b <= femur_left - 3]
        if medial:
            for a, b in medial:
                medial_mask[y, a:b] = 1
    # Отдельные тени/штрихи ниже таза не должны становиться нижним
    # ориентиром. Оставляем связный контур с опорой в ярком ядре кости.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(medial_mask)
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 5
                  and np.count_nonzero(core & (labels == i)) >= 20]
    if not candidates:
        raise ValueError("Не найден отдельный контур седалищной кости")
    # Самая яркая медиальная компонента может оказаться краем вертлужной
    # впадины выше седалищной кости, особенно после обрезания снизу.
    # Среди контуров с подтверждённым костным ядром нужен самый нижний.
    selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
    iy, ix = np.nonzero(labels == selected)
    bottom = int(iy.max())
    bottom_x = int(np.median(ix[iy == bottom]))
    if bottom <= top + 10:
        raise ValueError("Нарушено взаимное положение анатомических ориентиров")
    region = mask[top:bottom+1].copy()
    region[:, :int(shaft_x)] = False
    ry, rx = np.nonzero(region)
    lateral = int(rx.max())
    top_x = int(np.flatnonzero(outer[top])[0])
    def point(x, y):
        return [int(x if lateral_right else w-1-x), int(y)]
    return {"top": top * SPACING_Y, "bottom": (h-1-bottom) * SPACING_Y,
            "lateral": (w-1-lateral) * SPACING_X}, {
                "greater_trochanter": point(top_x, top),
                "ischium": point(bottom_x, bottom),
                "lateral_contour": point(lateral, top + int(ry[np.argmax(rx)])),
            }, "right" if lateral_right else "left"


def _clipped_inferior_fragment(gray):
    """Нижний изолированный фрагмент, устойчивый на трёх порогах.

    Проверка применяется при отказе хотя бы одного варианта поиска ориентиров.
    Ширина >=6 мм и площадь >=25 мм² исключают единичные пиксели/штрихи.
    Это геометрический признак неполного кадра, не диагноз перелома.
    """
    h, w = gray.shape
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    boxes = []
    for factor in (.3, .4, .5):
        _, _, stats, _ = cv2.connectedComponentsWithStats((smooth > threshold*factor).astype(np.uint8))
        candidates = [s for s in stats[1:] if s[4]*SPACING_X*SPACING_Y >= 25
                      and s[2]*SPACING_X >= 6 and s[1]+s[3] == h
                      and (h-s[1])*SPACING_Y < REQUIRED_MM["bottom"]]
        if len(candidates) != 1:
            return None
        boxes.append(candidates[0][:4])
    if np.ptp(np.array(boxes)[:, :2], axis=0).max() > 5:
        return None
    return list(map(int, boxes[1]))


def analyze_image(gray, side="auto"):
    """Измерения в исходных координатах; 3 порога не являются моделями ИИ."""
    if side not in ("auto", "left", "right"):
        raise ValueError("side должен быть auto, left или right")
    measurements, failures = [], []
    for factor in CONTOUR_FACTORS:
        try:
            measurements.append(_landmarks(gray, factor, side=side))
        except ValueError as exc:
            failures.append(str(exc))
    if failures:
        # Не приравниваем любой сбой сегментации к дефекту. Для этого
        # отдельного случая требуем видимый фрагмент кости, срезанный снизу,
        # и невозможность устойчиво найти полный набор ориентиров.
        fragment = _clipped_inferior_fragment(gray)
        if fragment is not None:
            return {"quality_class": 1, "processing_status": "Success",
                    "violation_type": ["incomplete_roi_at_bottom"],
                    "margins_mm": None, "required_mm": REQUIRED_MM.copy(),
                    "clipped_fragment_bbox_px": fragment,
                    "landmark_failures": failures,
                    "conclusion": "Область интереса неполна: нижняя граница кадра пересекает отдельный костный фрагмент; анатомические ориентиры не выделены"}
        raise ValueError("Не все варианты выделения контура успешны: " + "; ".join(sorted(set(failures))))
    if len({m[2] for m in measurements}) != 1:
        raise ValueError("Сегментации не согласованы по наружной стороне")
    intervals = {key: [min(m[0][key] for m in measurements), max(m[0][key] for m in measurements)]
                 for key in REQUIRED_MM}
    unstable = [key for key, (lo, hi) in intervals.items() if hi-lo > 6]
    # Допуск локализации в один пиксель; не ослабляет нормативные 30/20 мм.
    violations, uncertain = [], []
    for key, minimum in REQUIRED_MM.items():
        lo, hi = intervals[key]
        error = SPACING_X if key == "lateral" else SPACING_Y
        if hi < minimum:
            violations.append(key + "_margin_too_small")
        elif lo - error < minimum:
            uncertain.append(key)
    status = "Failure" if uncertain and not violations else "Success"
    return {"quality_class": 1 if violations else (None if uncertain else 0),
            "processing_status": status, "violation_type": violations,
            "margins_mm": {k: round(float(np.median([m[0][k] for m in measurements])), 2) for k in REQUIRED_MM},
            "margin_ranges_mm": intervals, "required_mm": REQUIRED_MM.copy(),
            "landmarks_px": measurements[1][1], "lateral_image_edge": measurements[1][2],
            "uncertain_margins": uncertain,
            "suspected_violation_type": [key + "_margin_borderline" for key in uncertain],
            "unstable_landmarks": unstable,
            "warnings": (["Положение части ориентиров зависит от порога сегментации; "
                          "измерения приблизительные, решение учитывает весь диапазон отступов"]
                         if unstable else []),
            "conclusion": ("Недостаточные отступы области интереса" if violations else
                           "Возможен дефект области интереса: отступ около порога, требуется ручная проверка" if uncertain else
                           "Отступы области интереса соответствуют ТЗ")}


def analyze(source, side="auto", image_data=None, model_path=DEFAULT_MODEL, device="cpu"):
    start = time.perf_counter()
    report = {"path_to_study": str(source), "study_uid": "", "image_uid": "",
              "anatomical_region": "unknown", "scope": "roi_scan_margins_only",
              "quality_class": None, "violation_type": [], "roi_check_performed": False,
              "pixel_spacing_mm": {"x": SPACING_X, "y": SPACING_Y}}
    try:
        if side not in ("auto", "left", "right"):
            raise ValueError("side должен быть auto, left или right")
        data = read_source(source) if image_data is None else image_data
        body = classify_body_part(data, model_path=model_path, device=device)
        report["body_part_classification"] = body
        if body["class"] == "spine":
            report.update(anatomical_region="spine", processing_status="Skipped",
                          conclusion="Определён позвоночник: проверка области интереса бедра неприменима")
            report["time_of_processing"] = round(time.perf_counter()-start, 4)
            return report
        if body["class"] not in ("hip_left", "hip_right"):
            raise ValueError("Неизвестный класс модели части тела")
        predicted_side = {"hip_left": "left", "hip_right": "right"}[body["class"]]
        selected_side = predicted_side if side == "auto" else side
        report.update(anatomical_region="proximal_femur", hip_side=selected_side,
                      predicted_hip_side=predicted_side,
                      hip_side_source="model" if side == "auto" else "argument",
                      expected_lateral_image_edge="right" if selected_side == "left" else "left")
        gray, meta = read_image(source, data=data)
        report.update(study_uid=meta["study_uid"], image_uid=meta["image_uid"])
        report["roi_check_performed"] = True
        report.update(analyze_image(gray, side=selected_side))
        # Независимая проверка геометрии не меняет сторону модели молча.
        # Она обнаруживает зеркальный экспорт или противоречие классификации.
        try:
            _, _, geometric_edge = _landmarks(gray, .12, side="auto")
        except ValueError:
            geometric_edge = None
        report["geometry_lateral_image_edge"] = geometric_edge
        report["side_geometry_consistent"] = (geometric_edge == report["expected_lateral_image_edge"]
                                               if geometric_edge is not None else None)
        if report["side_geometry_consistent"] is False:
            report.update(quality_class=None, processing_status="Failure", violation_type=[],
                          conclusion="Сторона бедра противоречит геометрии изображения: требуется ручная проверка",
                          error="Возможен зеркальный экспорт или ошибка определения стороны")
    except Exception as exc:
        report.update(quality_class=None, processing_status="Failure", violation_type=[],
                      conclusion="Требуется ручная проверка", error=str(exc))
    report["time_of_processing"] = round(time.perf_counter()-start, 4)
    return report


def read_manifest(path):
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1251")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    required = {"rel_path", "body_hip_right", "body_hip_left", "position_defects"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV не содержит необходимые поля")
    return rows


def resolve_image(rel_path, data_root=DEFAULT_ROOT):
    if urlparse(rel_path).scheme in ("http", "https", "file"):
        return rel_path
    path = Path(rel_path.replace("\\", "/"))
    if path.is_absolute() and path.is_file():
        return str(path)
    parts = path.parts
    candidates = [Path(data_root) / path]
    if "Исследования" in parts:
        candidates.append(Path(data_root).joinpath(*parts[parts.index("Исследования"):]))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Не найден снимок: {rel_path}; корень: {data_root}")


def evaluate(csv_path, data_root, limit=None, model_path=DEFAULT_MODEL, device="cpu"):
    """Оценка фиксированной эвристики. Не заменяет независимую валидацию."""
    reports, truth, predicted = [], [], []
    for row in read_manifest(csv_path)[:limit]:
        try:
            label = int(row["position_defects"])
            if label not in (0, 1):
                raise ValueError("Метка дефекта должна быть 0 или 1")
            source = resolve_image(row["rel_path"], data_root)
            # Сторона CSV не подаётся на вход — тот же конвейер, что в test.py.
            report = analyze(source, model_path=model_path, device=device)
            if report["quality_class"] is not None:
                truth.append(label)
                predicted.append(report["quality_class"])
        except Exception as exc:
            report = {"path_to_study": row["rel_path"], "quality_class": None,
                      "processing_status": "Failure", "error": str(exc)}
        report["reference_position_defects"] = row["position_defects"]
        reference_side = {("1", "0"): "left", ("0", "1"): "right"}.get(
            (row["body_hip_left"], row["body_hip_right"]))
        report["reference_hip_side"] = reference_side
        report["side_matches_reference"] = (report.get("predicted_hip_side") == reference_side
                                            if reference_side is not None and report.get("predicted_hip_side") else None)
        reports.append(report)
    tp = sum(t == p == 1 for t, p in zip(truth, predicted))
    tn = sum(t == p == 0 for t, p in zip(truth, predicted))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    sensitivity = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    positive_total = sum(x["reference_position_defects"] == "1" for x in reports)
    negative_total = sum(x["reference_position_defects"] == "0" for x in reports)
    return {"total": len(reports), "evaluated": len(truth), "manual_review": len(reports)-len(truth),
            "side_source": "body_part_model", "model_path": str(model_path),
            "side_compared": sum(r["side_matches_reference"] is not None for r in reports),
            "side_mismatches": sum(r["side_matches_reference"] is False for r in reports),
            "reference_defects_total": positive_total,
            "defects_detected_of_total": tp / positive_total if positive_total else None,
            "manual_review_by_reference": {"defect": positive_total-tp-fn,
                                            "normal": negative_total-tn-fp},
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "note": "confusion_matrix и sensitivity рассчитаны для обработанных строк. defects_detected_of_total учитывает все дефектные строки, включая ручную проверку. Датасет использовался при разработке: это не независимая валидация.",
            "results": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Оценить фиксированный CV-алгоритм на CSV")
    parser.add_argument("--csv", type=Path, default=HERE / "datasets/dataset_for_position.csv")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Веса body_part_model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.evaluate:
        parser.print_help()
        return 0
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit должен быть положительным")
    try:
        print(json.dumps(evaluate(args.csv, args.data_root, args.limit,
                                  model_path=args.model, device=args.device), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"processing_status": "Failure", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
