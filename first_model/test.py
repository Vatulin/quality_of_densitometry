import sys
import numpy as np
import pydicom
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["spine", "hip"]
MODEL_PATH = Path("best_model.pt")


def build_model():
    """Та же архитектура, что обучали."""
    model = models.resnet18(weights=None)          # веса загрузим из файла
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def pad_to_square(img):
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess(dcm_path: Path) -> torch.Tensor:
    """Точно тот же препроцессинг, что в DXADataset для val/test."""
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)

    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
    img = pad_to_square(img)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    x = torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0).unsqueeze(0) / 255.0
    return x


def predict_one(model, dcm_path: Path):
    x = preprocess(dcm_path).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return CLASS_NAMES[pred], probs


def main():
    if len(sys.argv) < 2:
        print("Использование: python predict.py <файл.dcm | папка>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        sys.exit(f"Не найдено: {target}")

    # загружаем модель
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # собираем список файлов
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.dcm"))

    if not files:
        sys.exit("DICOM-файлов не найдено")

    print(f"Модель: {MODEL_PATH}  |  устройство: {DEVICE}")
    print(f"Файлов: {len(files)}\n")

    for f in files:
        try:
            cls, probs = predict_one(model, f)
            print(f"{f.name:30s}  →  {cls:6s}  "
                  f"spine={probs[0]:.3f}  hip={probs[1]:.3f}")
        except Exception as e:
            print(f"{f.name:30s}  ОШИБКА: {e}")


if __name__ == "__main__":
    main()