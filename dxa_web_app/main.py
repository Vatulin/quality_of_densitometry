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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["spine", "hip"]
IMG_SIZE = 224

class SimpleModel:
    def __init__(self, weights_path):
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            print("Модель успешно загружена!")
        else:
            print("Файл best_model.pt не найден. ГГ.")
            
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, dcm_path: str):
        try:
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
            x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            self.model.eval()
            
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            del x
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                
            return {
                "status": "success",
                "class": CLASS_NAMES[int(probs.argmax())],
                "conf": f"{probs.max():.2%}"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

model = SimpleModel("best_model.pt")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

tasks = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return {"error": "загрузите ZIP-архив"}
    
    task_id = str(uuid.uuid4())
    zip_path = UPLOAD_DIR / f"{task_id}.zip"
    extract_dir = UPLOAD_DIR / task_id
    
    with open(zip_path, "wb") as f:
        f.write(await file.read())
        
    tasks[task_id] = {"status": "PROCESSING", "progress": 0, "total": 0, "results": []}
    
    background_tasks.add_task(process_zip, task_id, zip_path, extract_dir)
    
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        return {"error": "Задача не найдена"}
    return tasks[task_id]

@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "COMPLETED":
        return {"error": "Результат еще не готов"}
    
    df = pd.DataFrame(task["results"])
    csv_path = UPLOAD_DIR / f"{task_id}_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    return FileResponse(csv_path, filename=f"results_{task_id}.csv", media_type="text/csv")

def process_zip(task_id: str, zip_path: Path, extract_dir: Path):
    try:
        print(f"[{task_id}] Начинаем обработку архива...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        dcm_files = list(extract_dir.rglob("*.dcm")) + list(extract_dir.rglob("*.dicom"))
        tasks[task_id]["total"] = len(dcm_files)
        print(f"[{task_id}] Найдено файлов: {len(dcm_files)}")
        
        for i, dcm_file in enumerate(dcm_files):
            print(f"[{task_id}] Обрабатываем файл {i+1}/{len(dcm_files)}: {dcm_file.name}")
            
            pred = model.predict(str(dcm_file))
            
            rel_path = dcm_file.relative_to(extract_dir).as_posix()
            
            print(f"[{task_id}] Результат: {pred}")
            
            tasks[task_id]["results"].append({
                "filename": dcm_file.name,
                "path": rel_path,
                "prediction": pred.get("class", "N/A"),
                "confidence": pred.get("conf", "N/A"),
                "status": pred.get("status")
            })
            tasks[task_id]["progress"] = i + 1
            
        tasks[task_id]["status"] = "COMPLETED"
        print(f"[{task_id}] Обработка завершена!")
        
    except Exception as e:
        print(f"[{task_id}] ОШИБКА: {e}")
        tasks[task_id]["status"] = "FAILED"
        tasks[task_id]["error"] = str(e)