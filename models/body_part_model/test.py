import sys
import numpy as np
import pydicom
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models

# ---------- должно совпадать с train.py ----------
IMG_SIZE = 288
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["spine", "hip_right", "hip_left"]
MODEL_PATH = Path("best_model.pt")


def build_model():
    """Та же архитектура, что обучали: EfficientNet-B0 + 1-канальный вход + 3 класса."""
    model = models.efficientnet_b0(weights=None)

    # заменяем первый свёрточный слой на 1-канальный (как в train.py)
    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        1, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    model.features[0][0] = new_conv

    # финальный классификатор — 3 класса
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
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


def predict_one(model, dcm_path: Path, tta: bool = False):
    """Если tta=True — усредняем вероятности оригинала и лёгких сдвигов."""
    x = preprocess(dcm_path).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        if tta:
            x_right = torch.roll(x, shifts=8, dims=3)
            x_left  = torch.roll(x, shifts=-8, dims=3)
            probs = (probs
                     + torch.softmax(model(x_right), dim=1)
                     + torch.softmax(model(x_left),  dim=1)) / 3
        probs = probs[0].cpu().numpy()
    pred = int(probs.argmax())
    return CLASS_NAMES[pred], probs


def main():
    if len(sys.argv) < 2:
        print("Использование: python predict.py <файл.dcm | папка> [--tta]")
        sys.exit(1)

    target = Path(sys.argv[1])
    use_tta = "--tta" in sys.argv[2:]

    if not target.exists():
        sys.exit(f"Не найдено: {target}")

    # загружаем модель
    model = build_model().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # собираем список файлов
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.dcm"))

    if not files:
        sys.exit("DICOM-файлов не найдено")

    print(f"Модель: {MODEL_PATH}  |  устройство: {DEVICE}  |  TTA: {use_tta}")
    print(f"Файлов: {len(files)}\n")

    # считаем статистику по папке
    counts = {c: 0 for c in CLASS_NAMES}

    for f in files:
        try:
            cls, probs = predict_one(model, f, tta=use_tta)
            counts[cls] += 1
            print(f"{f.name:30s}  →  {cls:10s}  "
                  f"spine={probs[0]:.3f}  hip_r={probs[1]:.3f}  hip_l={probs[2]:.3f}")
        except Exception as e:
            print(f"{f.name:30s}  ОШИБКА: {e}")

    if len(files) > 1:
        print("\nИтог по папке:")
        for c in CLASS_NAMES:
            print(f"  {c:10s}: {counts[c]}")


if __name__ == "__main__":
    main()