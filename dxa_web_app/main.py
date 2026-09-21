import os
import uuid
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from torchvision import models
from typing import Dict, List
import cv2
from scipy.ndimage import gaussian_filter, median_filter
from scipy.special import expit

# --- НАСТРОЙКИ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
PIXEL_SPACING_X = 0.6  # мм (из ТЗ)
PIXEL_SPACING_Y = 1.05 # мм (из ТЗ)


class DICOMModel:
    def __init__(self, model_path: str, weights_path: str = None):
        self.model_path = Path(model_path)
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None
        self.device = DEVICE

    def preprocess_dcm(self, dcm_path: str) -> torch.Tensor:
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("L", (side, side), 0)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(img, copy=True)).float() / 255.0
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        return x

    def predict(self, dcm_path: str) -> Dict:
        raise NotImplementedError


class EfficientNetModel(DICOMModel):
    def __init__(self, model_path: str, weights_path: str, num_classes: int = 2):
        super().__init__(model_path, weights_path)
        self.model = models.efficientnet_b0(weights=None)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        if self.weights_path and self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state_dict_to_load = checkpoint

            if isinstance(checkpoint, dict):
                candidate_keys = ["encoder_state_dict", "model_state_dict", "state_dict", "model"]
                found = False
                for key in candidate_keys:
                    if key in checkpoint and isinstance(checkpoint[key], dict) and len(checkpoint[key]) > 0:
                        state_dict_to_load = checkpoint[key]
                        found = True
                        break
                
                if not found or len(state_dict_to_load) == 0:
                    state_dict_to_load = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}

                if isinstance(state_dict_to_load, dict) and len(state_dict_to_load) > 0:
                    first_key = list(state_dict_to_load.keys())[0]
                    if str(first_key).startswith("model."):
                        state_dict_to_load = {str(k).replace("model.", "", 1): v for k, v in state_dict_to_load.items()}

            try:
                self.model.load_state_dict(state_dict_to_load, strict=True)
            except RuntimeError:
                self.model.load_state_dict(state_dict_to_load, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, dcm_path: str) -> Dict:
        try:
            x = self.preprocess_dcm(dcm_path)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())
            confidence = float(probs.max())
            del x
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return {"status": "success", "class_id": pred_class, "confidence": confidence, "probabilities": probs.tolist()}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class SpineModel:
    """Специальный класс для spine_model (Random Forest + геометрические признаки)"""
    def __init__(self, model_path: str, weights_path: str):
        self.model_path = Path(model_path)
        self.weights_path = Path(weights_path)
        self.states = []
        self.threshold = 0.5
        self.device = DEVICE

        if self.weights_path.exists():
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            self.states = checkpoint.get("estimators", [])
            self.threshold = checkpoint.get("threshold", 0.5)
            print(f"✅ [spine_model] Загружен ансамбль Random Forest. Порог: {self.threshold:.4f}")
        else:
            print(f"⚠️ [spine_model] Веса не найдены: {self.weights_path}")

    def _geometry_features(self, img):
        """Точная копия функции из train.py"""
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
                score = ridge[0] - 0.1 * ((xs-width/2)/(width/2))**2
                back = np.zeros(ridge.shape, dtype=np.int64)
                for row in range(1, len(ridge)):
                    options = []
                    for delta in range(-2, 3):
                        previous = xs + delta
                        valid = (previous >= 0) & (previous < width)
                        options.append(np.where(valid, score[np.clip(previous, 0, width-1)] - 0.15 * delta**2, -np.inf))
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
        return result

    def _predict_estimator(self, state, x):
        n_geometry = state["n_geometry"]
        x_geom = np.asarray(x[:, :n_geometry], dtype=np.float32)
        predictions = []
        for tree in state["trees"]:
            left, right = tree["left"].numpy(), tree["right"].numpy()
            feature, threshold = tree["feature"].numpy(), tree["threshold"].numpy()
            nodes = np.zeros(len(x_geom), dtype=np.int64)
            while True:
                active = np.flatnonzero(left[nodes] != -1)
                if len(active) == 0:
                    break
                current = nodes[active]
                nodes[active] = np.where(x_geom[active, feature[current]] <= threshold[current], left[current], right[current])
            predictions.append(tree["probability"].numpy()[nodes])
        return np.mean(predictions, axis=0)

    def _predict_ensemble(self, x):
        return np.mean([self._predict_estimator(state, x) for state in self.states], axis=0)

    def predict(self, dcm_path: str) -> Dict:
        try:
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
            
            # Предобработка как в train.py
            side = max(img.size)
            square = Image.new("L", (side, side), 0)
            square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
            img = square.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)

            # Извлечение признаков и предсказание
            geom_features = self._geometry_features(img)
            x = geom_features[None, :]
            prob = float(self._predict_ensemble(x)[0])
            
            class_id = 1 if prob >= self.threshold else 0

            return {
                "status": "success",
                "class_id": class_id,
                "confidence": float(prob) if class_id == 1 else float(1.0 - prob),
                "probability": float(prob)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class PositionModel:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        print("✅ [position_model] Инициализирована (CV алгоритм)")

    def predict(self, dcm_path: str, body_part: str = "hip") -> Dict:
        try:
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr = (arr / arr.max() * 255).astype(np.uint8)
            
            violations = []
            confidence = 0.0
            smooth = cv2.GaussianBlur(arr, (3, 3), 0)
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

            return {"status": "success", "class_id": 1 if violations else 0, "confidence": confidence, "violations": violations}
        except Exception as e:
            return {"status": "error", "error": str(e)}


class MultiModelAnalyzer:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models = {}
        self._load_models()

    def _load_models(self):
        # Body Part Model
        bp_weights = self.models_dir / "body_part_model" / "weights" / "best_model.pt"
        if bp_weights.exists():
            self.models["body_part"] = EfficientNetModel(str(self.models_dir / "body_part_model"), str(bp_weights), num_classes=3)

        # Spine Model (Теперь используем специальный класс!)
        spine_weights = self.models_dir / "spine_model" / "weights" / "best_model.pt"
        if spine_weights.exists():
            self.models["spine"] = SpineModel(str(self.models_dir / "spine_model"), str(spine_weights))

        # Vertel Model
        vertel_weights = self.models_dir / "vertel_model" / "weights" / "best_model.pt"
        if vertel_weights.exists():
            self.models["verteb"] = EfficientNetModel(str(self.models_dir / "vertel_model"), str(vertel_weights), num_classes=3)

        # Artifact Model
        artifact_weights = self.models_dir / "artifact_model" / "weights" / "best_model.pt"
        if artifact_weights.exists():
            self.models["artifact"] = EfficientNetModel(str(self.models_dir / "artifact_model"), str(artifact_weights), num_classes=2)

        # Position Model
        position_path = self.models_dir / "position_model"
        if position_path.exists():
            self.models["position"] = PositionModel(str(position_path))

        print(f"✅ Всего успешно загружено моделей: {len(self.models)}")

    def analyze(self, dcm_path: str) -> Dict:
        results = {
            "filename": Path(dcm_path).name,
            "anatomical_region": None,
            "quality_class": 0,
            "quality_prob": 0.0,
            "violation_type": [],
            "model_predictions": {}
        }

        # 1. Определение части тела
        if "body_part" in self.models:
            pred = self.models["body_part"].predict(dcm_path)
            results["model_predictions"]["body_part"] = pred
            if pred["status"] == "success":
                class_id = pred["class_id"]
                if class_id == 0:
                    results["anatomical_region"] = "Поясничный отдел позвоночника"
                else:
                    results["anatomical_region"] = "Проксимальный отдел бедра"

        violations = []
        max_confidence = 0.0
        region = results["anatomical_region"]

        # 2. Анализ позвоночника
        if region == "Поясничный отдел позвоночника":
            if "spine" in self.models:
                pred = self.models["spine"].predict(dcm_path)
                results["model_predictions"]["spine"] = pred
                if pred["status"] == "success" and pred["class_id"] == 1:
                    violations.append("Не выравнена ось позвоночника")
                    violations.append("Присутствуют посторонние предметы")
                    max_confidence = max(max_confidence, pred["confidence"])
            
            if "verteb" in self.models:
                pred = self.models["verteb"].predict(dcm_path)
                results["model_predictions"]["verteb"] = pred
                if pred["status"] == "success" and pred["class_id"] > 0:
                    if "Не выравнена ось позвоночника" not in violations:
                        violations.append("Не выравнена ось позвоночника")
                    max_confidence = max(max_confidence, pred["confidence"])

        # 3. Анализ артефактов
        if "artifact" in self.models:
            pred = self.models["artifact"].predict(dcm_path)
            results["model_predictions"]["artifact"] = pred
            if pred["status"] == "success" and pred["class_id"] > 0:
                if "Некорректная укладка" not in violations:
                    violations.append("Некорректная укладка")
                max_confidence = max(max_confidence, pred["confidence"])

        # 4. Анализ позиции (для бедра)
        if "position" in self.models and region == "Проксимальный отдел бедра":
            pred = self.models["position"].predict(dcm_path, body_part="hip")
            results["model_predictions"]["position"] = pred
            if pred["status"] == "success" and pred["class_id"] == 1:
                for v in pred.get("violations", []):
                    if v not in violations:
                        violations.append(v)
                max_confidence = max(max_confidence, pred["confidence"])

        results["violation_type"] = violations
        results["quality_prob"] = float(max_confidence)
        results["quality_class"] = 1 if violations else 0

        return results


# --- FASTAPI ПРИЛОЖЕНИЕ ---
app = FastAPI(title="DXA Multi-Model Analyzer")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analyzer = None
tasks = {}

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
    if file.filename.endswith(".zip"):
        background_tasks.add_task(process_zip_with_models, task_id, file_path)
    else:
        background_tasks.add_task(process_single_file, task_id, file_path)
    return {"task_id": task_id}

def process_single_file(task_id: str, file_path: Path):
    try:
        if analyzer is None: raise Exception("Анализатор не инициализирован")
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
        if analyzer is None: raise Exception("Анализатор не инициализирован")
        extract_dir = UPLOAD_DIR / task_id
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
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
    rows = []
    for result in task["results"]:
        rows.append({
            "filename": result["filename"],
            "path": result.get("path", ""),
            "anatomical_region": result["anatomical_region"] or "",
            "quality_class": result["quality_class"],
            "quality_prob": result["quality_prob"],
            "violation_type": ";".join(result["violation_type"]) if result["violation_type"] else ""
        })
    df = pd.DataFrame(rows)
    csv_path = UPLOAD_DIR / f"{task_id}_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return FileResponse(csv_path, filename=f"results_{task_id}.csv", media_type="text/csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)