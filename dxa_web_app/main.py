"""API загрузки DICOM, оркестрация анализа и экспорт отчёта."""

import uuid
import zipfile
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd
import pydicom
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

if __package__:
    from .visualization.service import register_visualization
    from .inference_models import (
        BodyPartModel, SpineModel, HipQualityModel, ArtifactModel,
        SpinePositionModel, PositionModel,
    )
else:
    from visualization.service import register_visualization
    from inference_models import (
        BodyPartModel, SpineModel, HipQualityModel, ArtifactModel,
        SpinePositionModel, PositionModel,
    )


class MultiModelAnalyzer:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models: Dict[str, object] = {}
        self.model_load_errors: Dict[str, str] = {}
        self._load_models()

    def _try_load(self, name: str, cls, *paths) -> bool:
        for p in paths:
            p = Path(p)
            if p.exists():
                try:
                    self.models[name] = cls(str(p))
                    return True
                except Exception as e:
                    self.model_load_errors[name] = f"{type(e).__name__}: {e}"
                    print(f"⚠️ [{name}] не удалось загрузить {p}: {e}")
        self.model_load_errors.setdefault(name, "Файл модели или алгоритма не найден")
        print(f"⚠️ [{name}] модель или алгоритм недоступны: {[str(p) for p in paths]}")
        return False

    def _load_models(self):
        md = self.models_dir

        # 1) Body part — EfficientNet-B0: spine / hip_right / hip_left
        self._try_load(
            "body_part", BodyPartModel,
            md / "body_part_model" / "weights" / "best_model.pt",
        )

        # 2) Spine — ансамбль (RF или LogReg) + опционально CNN
        self._try_load(
            "spine", SpineModel,
            md / "spine_model" / "weights" / "best_model.pt",
        )

        self._try_load(
            "spine_position", SpinePositionModel,
            md / "spine_position_model" / "final_spine_model.pth",
        )

        # 3) Hip quality (vertel) — EfficientNet_B0 + side-embedding
        self._try_load(
            "hip_quality", HipQualityModel,
            md / "vertel_model" / "weights" / "best_hip_quality.pt",
        )

        # 4) Artifacts — timm.resnet18, sigmoid
        self._try_load(
            "artifact", ArtifactModel,
            md / "artifact_model" / "weights" / "best_artifact_model.pth",
        )

        # 5) Position — CV, без весов
        self._try_load("position", PositionModel, md / "position_model" / "train.py")

        print(f"✅ Всего успешно загружено моделей: {len(self.models)}")

    def _predict(self, name: str, dcm_path: str, **kwargs) -> Dict:
        if name not in self.models:
            reason = self.model_load_errors.get(name, "Модель не загружена")
            raise RuntimeError(f"{name}: {reason}")
        try:
            prediction = self.models[name].predict(dcm_path, **kwargs)
            if not isinstance(prediction, dict):
                raise ValueError("Модель вернула некорректный результат")
            if prediction.get("status") != "success":
                raise RuntimeError(prediction.get("error") or "Ошибка модели без описания")
            return prediction
        except Exception as e:
            raise RuntimeError(f"{name}: {e}") from e

    def analyze(self, dcm_path: str) -> Dict:
        started = perf_counter()
        metadata = {"study_uid": "", "image_uid": ""}
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            metadata = {
                "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                "image_uid": str(getattr(ds, "SOPInstanceUID", "")),
            }
            result = self._analyze(dcm_path)
        except Exception as e:
            result = failed_result(dcm_path, e)
        result.update(metadata)
        result["path_to_study"] = str(dcm_path)
        result["time_of_processing"] = round(perf_counter() - started, 6)
        return result

    def _analyze(self, dcm_path: str) -> Dict:
        results = {
            "filename": Path(dcm_path).name,
            "path": str(dcm_path),
            "processing_status": "Success",
            "error_message": "",
            "anatomical_region": None,
            "hip_side": None,
            "quality_class": 0,
            "quality_prob": 0.0,
            "violation_prob": 0.0,     
            "violation_type": [],
            "model_predictions": {},
        }

        # 1) Часть тела и сторона
        hip_side = None
        pred = self._predict("body_part", dcm_path)
        results["model_predictions"]["body_part"] = pred
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
        if region is None:
            raise ValueError("body_part: неизвестная анатомическая область")
        violations: List[str] = []
        violation_probs: List[float] = []   

        if region == "Поясничный отдел позвоночника":
            checks = [
                ("spine", {}, "Не выравнена ось позвоночника"),
                ("spine_position", {}, "Некорректная укладка позвоночника"),
            ]
        else:
            checks = [("hip_quality", {"side": hip_side}, None)]
        checks.append(("artifact", {}, "Наличие артефактов"))
        if hip_side is not None:
            checks.append(("position", {"side": hip_side}, None))

        roi_violation = False
        for model_name, kwargs, violation_label in checks:
            pred = self._predict(model_name, dcm_path, **kwargs)
            results["model_predictions"][model_name] = pred
            if "probability" in pred:
                violation_probs.append(float(pred["probability"]))
            violation_probs.extend(float(p) for p in pred.get("probabilities", []))
            if pred["class_id"] == 1:
                labels = [violation_label] if violation_label else pred.get("violations", [])
                for label in labels:
                    if label not in violations:
                        violations.append(label)
                roi_violation |= model_name == "position"

        max_violation_prob = max(violation_probs) if violation_probs else 0.0

        results["violation_type"] = violations
        results["violation_prob"] = float(max_violation_prob)

        if violations:
            results["quality_class"] = 1
            # CV определяет нарушение без вероятности; не подменяем её оценкой другой модели.
            results["quality_prob"] = None if roi_violation else float(max_violation_prob)
        else:
            results["quality_class"] = 0
            results["quality_prob"] = float(1.0 - max_violation_prob)

        return results


def failed_result(dcm_path: str, error: Exception) -> Dict:
    """Техническая ошибка не является оценкой качества изображения."""
    return {
        "filename": Path(dcm_path).name,
        "path": str(dcm_path),
        "path_to_study": str(dcm_path),
        "study_uid": "",
        "image_uid": "",
        "time_of_processing": 0.0,
        "anatomical_region": None,
        "hip_side": None,
        "quality_class": None,
        "quality_prob": None,
        "violation_prob": None,
        "violation_type": [],
        "model_predictions": {},
        "processing_status": "Failure",
        "error_message": f"{type(error).__name__}: {error}",
    }


# FASTAPI 

app = FastAPI(title="DXA Multi-Model Analyzer")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analyzer: MultiModelAnalyzer | None = None
tasks: Dict = {}
register_visualization(app, tasks, lambda: analyzer)


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
async def analyze_file(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] | None = File(None),
    file: UploadFile | None = File(None),
):
    uploads = list(files or [])
    if file is not None:  # Совместимость с прежними клиентами API.
        uploads.append(file)
    if not uploads or any(not upload.filename for upload in uploads):
        return {"error": "Выберите файлы DICOM или ZIP-архивы"}
    task_id = str(uuid.uuid4())
    saved_files = []
    for index, upload in enumerate(uploads, 1):
        filename = Path(upload.filename.replace("\\", "/")).name
        if filename in {"", ".", ".."}:
            return {"error": "Некорректное имя файла"}
        directory = UPLOAD_DIR / task_id / str(index)
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / filename
        with file_path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                output.write(chunk)
        await upload.close()
        saved_files.append(file_path)
    tasks[task_id] = {"status": "PROCESSING", "progress": 0, "total": 0, "results": []}
    background_tasks.add_task(process_files, task_id, saved_files)
    return {"task_id": task_id}


def analyze_safely(file_path: Path) -> Dict:
    started = perf_counter()
    try:
        if analyzer is None:
            raise Exception("Анализатор не инициализирован")
        return analyzer.analyze(str(file_path))
    except Exception as e:
        result = failed_result(str(file_path), e)
        result["time_of_processing"] = round(perf_counter() - started, 6)
        return result


def process_files(task_id: str, file_paths: List[Path]):
    task = tasks[task_id]
    pending = []
    for file_path in file_paths:
        started = perf_counter()
        try:
            if file_path.suffix.lower() == ".zip":
                extract_dir = file_path.parent / (file_path.name + "_contents")
                with zipfile.ZipFile(file_path, "r") as archive:
                    archive.extractall(extract_dir)
                images = sorted(p for p in extract_dir.rglob("*")
                                if p.is_file() and p.suffix.lower() in {".dcm", ".dicom"})
                if not images:
                    raise ValueError("В архиве нет DICOM-файлов (.dcm, .dicom)")
                pending.extend((p, None) for p in images)
            else:
                pending.append((file_path, None))
        except Exception as e:
            result = failed_result(str(file_path), e)
            result["time_of_processing"] = round(perf_counter() - started, 6)
            pending.append((file_path, result))
    task["total"] = len(pending)
    task["results"] = []
    for file_path, failure in pending:
        result = failure if failure is not None else analyze_safely(file_path)
        result["path_to_study"] = str(file_path)
        result["path"] = str(file_path)
        task["results"].append(result)
        task["progress"] += 1
    task["status"] = "COMPLETED"


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"error": "Задача не найдена"})


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "COMPLETED":
        return {"error": "Результат еще не готов"}

    columns = ["path_to_study", "study_uid", "image_uid", "anatomical_region",
               "quality_class", "violation_type", "processing_status", "time_of_processing",
               "error_message"]
    rows = []
    for result in task["results"]:
        rows.append({
            "path_to_study": result["path_to_study"],
            "study_uid": result["study_uid"],
            "image_uid": result["image_uid"],
            "anatomical_region": result["anatomical_region"] or "",
            "quality_class": result["quality_class"],
            "violation_type": ";".join(result["violation_type"]) if result["violation_type"] else "",
            "processing_status": result["processing_status"],
            "time_of_processing": result["time_of_processing"],
            "error_message": result["error_message"],
        })
    df = pd.DataFrame(rows, columns=columns)
    df["quality_class"] = pd.array(df["quality_class"], dtype="Int64")
    csv_path = UPLOAD_DIR / f"{task_id}_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return FileResponse(csv_path, filename=f"results_{task_id}.csv", media_type="text/csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
