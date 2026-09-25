"""On-demand explanations and an in-memory DICOM Secondary Capture series."""
import base64
import json
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from threading import Lock
from tempfile import SpooledTemporaryFile
import zipfile

import cv2
import numpy as np
import pydicom
import torch
from PIL import Image
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
from pydicom.dataset import FileDataset, FileMetaDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter


def jet_colors(values):
    return cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def spine_axis_overlay(rgb, arr):
    """Trace a continuous centre ridge with a smooth dynamic-programming path.

    Coordinates are restored from the padded image before drawing. This is a
    geometric illustration, not a vertebral segmentation or a new diagnosis.
    """
    size = 224
    image = np.asarray(square(arr, size), dtype=np.float64) / 255
    y0, y1, x0, x1 = int(.12*size), int(.82*size), int(.2*size), int(.8*size)
    # Blur across both cortical edges so the path follows the centre of the
    # vertebral bodies instead of zigzagging along an individual bright edge.
    ridge = (gaussian_filter(image, (8, 12)) - gaussian_filter(image, (8, 36)))[y0:y1, x0:x1]
    if float(ridge.std()) < 1e-6:
        raise ValueError("Недостаточно контраста для выделения оси позвоночника")
    ridge /= ridge.std()
    width = ridge.shape[1]
    xs = np.arange(width)
    score = ridge[0] - .1 * ((xs-width/2)/(width/2))**2
    back = np.zeros(ridge.shape, dtype=np.int32)
    for row in range(1, len(ridge)):
        options = []
        for delta in range(-2, 3):
            previous = xs+delta
            valid = (previous >= 0) & (previous < width)
            options.append(np.where(valid, score[np.clip(previous, 0, width-1)]-.15*delta**2, -np.inf))
        options = np.asarray(options)
        best = options.argmax(axis=0)
        back[row] = np.clip(xs+best-2, 0, width-1)
        score = ridge[row] + options[best, xs]
    path = np.zeros(len(ridge), dtype=np.int32)
    path[-1] = score.argmax()
    for row in range(len(ridge)-1, 0, -1):
        path[row-1] = back[row, path[row]]
    xx = gaussian_filter1d(median_filter(path.astype(float), size=7), 8) + x0
    h, w = rgb.shape[:2]
    side = max(h, w)
    # PIL resize maps pixel centres, including integer padding offsets.
    xx = (xx+.5)*side/size-.5-(side-w)//2
    yy = (np.arange(y0, y1)+.5)*side/size-.5-(side-h)//2
    valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    xx, yy = xx[valid], yy[valid]
    if len(xx) < 12:
        raise ValueError("Не удалось проследить ось позвоночника")
    points = np.rint(np.column_stack((xx, yy))).astype(np.int32)
    reference_x = int(round(float(np.median(xx))))
    overlay = rgb.copy()
    thickness = max(2, round(min(h, w)/160))
    cv2.line(overlay, (reference_x, int(points[0, 1])), (reference_x, int(points[-1, 1])),
             (34, 211, 238), thickness+2, cv2.LINE_AA)
    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], False, (255, 153, 51), thickness, cv2.LINE_AA)
    return overlay


def png(rgb):
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def square(arr, size):
    image = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    w, h = image.size
    canvas = Image.new("L", (max(w, h), max(w, h)), 0)
    canvas.paste(image, ((max(w, h)-w)//2, (max(w, h)-h)//2))
    return canvas.resize((size, size), Image.Resampling.BILINEAR)


def restore_map(heat, shape, padded):
    h, w = shape
    if padded:
        side = max(h, w)
        heat = cv2.resize(heat, (side, side))
        heat = heat[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w]
    else:
        heat = cv2.resize(heat, (w, h))
    if not np.isfinite(heat).all() or heat.max() <= 1e-12:
        raise ValueError("Модель не дала устойчивой карты влияния для этого снимка")
    return np.clip(heat / heat.max(), 0, 1)


def heatmap(model, name, path, arr, target, side):
    """Use the exact classifier preprocessing; never invent an anatomical mask."""
    if name == "hip_quality":
        pixels = np.array(square(arr, model.img_size), copy=True).astype(np.float32)/255
        x = torch.from_numpy(pixels)[None, None].to(model.device)
        x = ((x-.5)/.25).requires_grad_(True)
    else:
        pixels = cv2.resize(model._read_dxa(path), (model.IMG_SIZE, model.IMG_SIZE)).astype(np.float32)/255
        pixels = (pixels-model.MEAN)/model.STD
        x = torch.from_numpy(pixels.transpose(2, 0, 1)).float()[None].to(model.device).requires_grad_(True)
    with torch.enable_grad():
        if name == "hip_quality":
            logits = model.model(x, torch.tensor([0 if side == "right" else 1], device=model.device))
        else:
            logits = model.model(x)
        gradient, = torch.autograd.grad(logits.reshape(-1)[target], x)
        heat = gradient.detach().abs().mean(dim=1)[0].cpu().numpy()
    heat = cv2.GaussianBlur(heat, (0, 0), 3)
    return restore_map(heat, arr.shape, name == "hip_quality"), "Чувствительность выхода модели к пикселям (градиент). Тёплые цвета — большее влияние; это не точная граница нарушения."


def roi_overlay(rgb, report):
    image = rgb.copy()
    h, w = image.shape[:2]
    points = report.get("landmarks_px", {})
    for key, label, margin in (("greater_trochanter", "1", "top"), ("ischium", "2", "bottom"), ("lateral_contour", "3", "lateral")):
        if key not in points:
            continue
        x, y = map(int, points[key])
        codes = report.get("violation_type", [])
        color = (255, 70, 70) if any(c.startswith(margin+"_margin") for c in codes) else (60, 220, 130)
        end = (x, 0) if margin == "top" else (x, h-1) if margin == "bottom" else (w-1 if report.get("lateral_image_edge") == "right" else 0, y)
        distance = float(np.hypot(end[0]-x, end[1]-y))
        if distance > 0:
            tip = min(.4, 7.0/distance)
            cv2.arrowedLine(image, (x, y), end, color, 2, cv2.LINE_AA, tipLength=tip)
            cv2.arrowedLine(image, end, (x, y), color, 2, cv2.LINE_AA, tipLength=tip)
        else:
            # A landmark on the border has zero margin: point to the border
            # from inside the frame rather than inventing a measurement span.
            start = (x, min(h-1, 16)) if margin == "top" else (x, max(0, h-17)) if margin == "bottom" else (max(0, w-17) if end[0] == w-1 else min(w-1, 16), y)
            cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=.4)
        cv2.putText(image, label, (max(0, min(x+7, w-18)), max(18, y)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    box = report.get("clipped_fragment_bbox_px")
    if box:
        x, y, bw, bh = map(int, box)
        cv2.rectangle(image, (x, y), (x+bw-1, y+bh-1), (255, 70, 70), 2)
        # The clipped-fragment fallback has no landmarks. Mark the actual
        # bottom border instead of silently returning only a rectangle.
        centre = max(0, min(w-1, x+bw//2))
        cv2.arrowedLine(image, (centre, max(0, min(y, h-17))),
                        (centre, h-1), (255, 70, 70), 2, cv2.LINE_AA, tipLength=.3)
    if not points and not box:
        raise ValueError("Координаты ориентиров отсутствуют")
    return image


def secondary_capture(source, rgb, series_uid, number, title):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0"*128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = source.StudyInstanceUID
    ds.SeriesInstanceUID = series_uid
    for key in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime", "StudyID", "AccessionNumber"):
        setattr(ds, key, source.get(key, ""))
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.Modality = "OT"
    ds.SeriesNumber = 900
    ds.InstanceNumber = number
    ds.SeriesDescription = "DXA quality visualization"
    ds.DerivationDescription = title
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ConversionType = "WSD"
    ds.BurnedInAnnotation = "YES"
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S")
    ref = Dataset()
    ref.ReferencedSOPClassUID = source.SOPClassUID
    ref.ReferencedSOPInstanceUID = source.SOPInstanceUID
    ds.SourceImageSequence = [ref]
    ds.Rows, ds.Columns = rgb.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = rgb.astype(np.uint8).tobytes()
    stream = BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def build(result, analyzer):
    path = result["path_to_study"]
    ds = pydicom.dcmread(path)
    raw = ds.pixel_array.astype(np.float32)
    if raw.ndim != 2 or int(ds.get("SamplesPerPixel", 1)) != 1 or not np.isfinite(raw).all():
        raise ValueError("Визуализация поддерживает однокадровые монохромные изображения")
    arr = (raw-raw.min())/max(float(np.ptp(raw)), 1e-8)
    display = 1-arr if ds.get("PhotometricInterpretation") == "MONOCHROME1" else arr
    rgb = np.repeat((display*255).astype(np.uint8)[..., None], 3, axis=2)
    payload = {"original": png(rgb), "items": [], "warnings": []}
    frames = []
    predictions = result.get("model_predictions", {})
    roi_prediction = predictions.get("position", {})
    roi_report = roi_prediction.get("roi_details") or {}
    roi_available = (roi_prediction.get("status") == "success"
                     and roi_report.get("processing_status") == "Success")
    roi_rendered = False
    titles = {"artifact": "Наличие артефактов", "spine": "Не выравнена ось позвоночника", "spine_position": "Некорректная укладка позвоночника", "position": "Отступы ROI"}
    for name, pred in sorted(predictions.items(), key=lambda pair: pair[0] == "position"):
        # Measurements are useful even when the separate ROI classifier flags
        # a defect but the margin algorithm finds all distances sufficient.
        if name == "position" and roi_rendered:
            continue
        if name == "body_part" or (pred.get("class_id") != 1 and not (name == "position" and roi_available)):
            continue
        targets = [(0, titles.get(name, name))]
        if name == "hip_quality":
            model = analyzer.models.get(name)
            targets = [(j, model.LABEL_RU[label]) for j, label in enumerate(model.LABEL_COLS) if model.LABEL_RU[label] in pred.get("violations", [])] if model else [(0, "Качество бедра")]
        for target, title in targets:
            try:
                details = pred.get("roi_details")
                hip_roi = (name == "hip_quality" and model is not None
                           and model.LABEL_COLS[target] == "roi_correctness")
                kind = "heatmap"
                legend = None
                if name == "position" or hip_roi:
                    if hip_roi:
                        if not roi_available:
                            raise ValueError("Измерения отступов недоступны: стрелки без анатомических ориентиров не строятся")
                        details = roi_report
                    kind = "roi"
                    overlay = roi_overlay(rgb, details or {})
                    explanation = "1 — большой вертел; 2 — седалищная кость; 3 — наружный контур. Двусторонние стрелки показывают отступы до краёв снимка: красные — недостаточные или пограничные, зелёные — достаточные. При нулевом отступе одиночная стрелка указывает на край снимка."
                    if hip_roi:
                        explanation += " Классификатор ROI и проверка отступов оценивают разные признаки: зелёные стрелки не отменяют заключение классификатора."
                    if (details or {}).get("clipped_fragment_bbox_px"):
                        explanation += " Стрелка у нижнего края показывает обрезанный фрагмент; отступы без выделенных ориентиров не измерены."
                    roi_rendered = True
                elif name == "spine":
                    kind = "spine_axis"
                    overlay = spine_axis_overlay(rgb, display)
                    explanation = "Голубая линия — вертикальный ориентир; оранжевая — автоматически прослеженная ось позвоночника. Ось приблизительная, построена по изображению для пояснения результата модели."
                else:
                    heat, explanation = heatmap(analyzer.models[name], name, path, arr, target, result.get("hip_side"))
                    colors = jet_colors((heat*255).astype(np.uint8))
                    legend = png(jet_colors(np.arange(256, dtype=np.uint8)[None, :]))
                    alpha = (heat*.65)[..., None]
                    overlay = (rgb*(1-alpha)+colors*alpha).astype(np.uint8)
                payload["items"].append({"title": title, "image": png(overlay), "explanation": explanation,
                                         "details": details, "kind": kind, "legend": legend})
                frames.append((title, overlay))
            except Exception as exc:
                payload["warnings"].append(f"{title}: визуализация недоступна — {exc}")
    archive = BytesIO()
    if frames:
        series_uid = generate_uid()
        try:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
                for number, (title, frame) in enumerate(frames, 1):
                    output.writestr(f"visualization_{number}.dcm", secondary_capture(ds, frame, series_uid, number, title))
            payload["series_uid"] = series_uid
        except Exception as exc:
            payload["warnings"].append(f"Не удалось сформировать DICOM-серию: {exc}")
            archive = BytesIO()
    return payload, archive.getvalue()


def register_visualization(app, tasks, get_analyzer):
    cache = OrderedDict()
    lock = Lock()

    def retrieve(task_id, index):
        task = tasks.get(task_id)
        if task is None or index < 0 or index >= len(task.get("results", [])):
            raise HTTPException(404, "Исследование не найдено")
        key = (task_id, index)
        with lock:
            if key not in cache:
                try:
                    cache[key] = build(task["results"][index], get_analyzer())
                except Exception as exc:
                    raise HTTPException(422, f"Не удалось визуализировать снимок: {exc}") from exc
                while len(cache) > 8:
                    cache.popitem(last=False)
            cache.move_to_end(key)
            return cache[key]

    @app.get("/api/visualization/{task_id}/{index}")
    def visualization(task_id: str, index: int):
        return retrieve(task_id, index)[0]

    @app.get("/api/visualization/{task_id}/{index}/dicom")
    def download_series(task_id: str, index: int):
        _, archive = retrieve(task_id, index)
        if not archive:
            raise HTTPException(409, "Дополнительная серия отсутствует")
        return Response(archive, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="visualization_series.zip"'})

    @app.get("/api/series/{task_id}")
    def download_batch_series(task_id: str):
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Задача не найдена")
        if task.get("status") != "COMPLETED":
            raise HTTPException(409, "Дождитесь завершения анализа")

        # Large batches spill to disk instead of retaining the entire ZIP in RAM.
        stream = SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b")
        manifest = []
        try:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as output:
                for index, result in enumerate(list(task["results"])):
                    entry = {
                        "index": index, "filename": result.get("filename", ""),
                        "path_to_study": result.get("path_to_study", ""),
                        "study_uid": result.get("study_uid", ""),
                        "image_uid": result.get("image_uid", ""),
                        "processing_status": result.get("processing_status"),
                        "files": [], "warnings": [],
                    }
                    try:
                        if result.get("processing_status") != "Success":
                            raise ValueError(result.get("error_message") or "Ошибка анализа изображения")
                        payload, archive = retrieve(task_id, index)
                        entry["warnings"] = payload.get("warnings", [])
                        if not archive:
                            entry.update(
                                status="Failure" if entry["warnings"] else "Skipped",
                                reason="; ".join(entry["warnings"]) or "Дополнительная визуальная серия отсутствует",
                            )
                        else:
                            # Read the whole individual series before adding it, so a
                            # corrupt member cannot leave a half-exported series.
                            with zipfile.ZipFile(BytesIO(archive)) as source:
                                members = [(item.filename, source.read(item))
                                           for item in source.infolist() if not item.is_dir()]
                            if not members:
                                raise ValueError("Дополнительная серия пуста")
                            for number, (_, data) in enumerate(members, 1):
                                name = f"image_{index + 1:04d}/visualization_{number}.dcm"
                                output.writestr(name, data)
                                entry["files"].append(name)
                            entry.update(status="Partial" if entry["warnings"] else "Success",
                                         series_uid=payload.get("series_uid", ""))
                    except Exception as exc:
                        entry.update(status="Failure", reason=str(exc.detail if isinstance(exc, HTTPException) else exc))
                    manifest.append(entry)
                output.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            stream.seek(0)
        except Exception:
            stream.close()
            raise

        def chunks():
            try:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
            finally:
                stream.close()

        return StreamingResponse(chunks(), media_type="application/zip",
                                 headers={"Content-Disposition": 'attachment; filename="visualization_series.zip"'},
                                 background=BackgroundTask(stream.close))
