"""Классификация части тела, затем CV-проверка отступов ROI бедра.

Пример: python -B test.py ../../../Data/Тест/CR000001_ЛПОБ.dcm --side left
Прямая HTTP(S)-ссылка должна возвращать снимок, а не страницу облачного диска.
Вывод JSON (по умолчанию) или CSV в stdout; новых файлов нет.
Сначала first_model/best_model.pt определяет spine/hip_right/hip_left.
Для spine проверка отступов бедра пропускается. Сторона бедра берётся
из предсказания модели и управляет поиском наружного края. --side позволяет
явно переопределить сторону; противоречие геометрии требует ручной проверки.
Код выхода: 0 — отступы соблюдены/неприменимо, 1 — нарушение, 2 — ошибка.
Только исходный масштаб 0.6 мм по X, 1.05 мм по Y и вертикальная укладка.
Зависимости: numpy, opencv-python, pydicom, torch, torchvision, pillow.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
from pathlib import Path

if __package__:
    from .train import analyze, resolve_image, DEFAULT_MODEL
else:
    from quality_of_densitometry.models.position_model.train import analyze, resolve_image, DEFAULT_MODEL

def predict(source, side="auto", model_path=DEFAULT_MODEL, device="cpu"):
    """Тот же конвейер first_model -> CV, что при оценке датасета в train.py."""
    return analyze(source, side=side, model_path=model_path, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Путь, rel_path из датасета, file:// или прямая HTTP(S)-ссылка")
    parser.add_argument("--side", choices=("auto", "left", "right"), default="auto",
                        help="Сторона для поиска ROI: auto — из first_model; left/right — явное указание")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Веса модели spine/hip_right/hip_left")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    source = args.source
    try:
        source = resolve_image(source)
    except (ValueError, OSError):
        pass  # analyze формирует единый отчёт об ошибке чтения.
    result = predict(source, args.side, args.model, args.device)
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(result), lineterminator="\n")
        writer.writeheader()
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in result.items()})
    if result["processing_status"] == "Skipped":
        return 0
    return 2 if result["quality_class"] is None else result["quality_class"]


if __name__ == "__main__":
    raise SystemExit(main())
