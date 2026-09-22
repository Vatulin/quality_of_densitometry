"""Каскад body_part_model -> spine_model для одного DICOM или папки.

python test.py "путь/к/снимку.dcm"
python test.py "путь/к/папке"
Из корня проекта: python src/quality_of_densitometry/models/spine_model/test.py "путь/к/снимку.dcm"
Для каждого снимка позвоночника выводится только «Да» (>5°) или «Нет» (<=5°).
Если определено бедро или произошла ошибка, сообщение выводится в stderr.
body_part_model различает spine/hip_right/hip_left, а не любые посторонние изображения.
"""

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import torch

if __package__:
    from ..body_part_model.test import CLASS_NAMES, build_model as build_body_model, preprocess as preprocess_body
    from .train import DEFAULT_WEIGHTS, HERE, PREPROCESS, FeaturePredictor, build_model, image_tensor, read_image
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from quality_of_densitometry.models.body_part_model.test import CLASS_NAMES, build_model as build_body_model, preprocess as preprocess_body
    from quality_of_densitometry.models.spine_model.train import DEFAULT_WEIGHTS, HERE, PREPROCESS, FeaturePredictor, build_model, image_tensor, read_image


DEFAULT_FIRST_WEIGHTS = HERE.parent / "body_part_model" / "weights" / "best_model.pt"


class SpinePipeline:
    def __init__(self, first_weights=DEFAULT_FIRST_WEIGHTS,
                 spine_weights=DEFAULT_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.first = build_body_model().to(self.device)
        self.first.load_state_dict(torch.load(first_weights, map_location="cpu", weights_only=True))
        self.first.eval()
        self.spine_weights = Path(spine_weights)
        self.spine = None
        self.threshold = None

    def _load_spine(self):
        checkpoint = torch.load(self.spine_weights, map_location="cpu", weights_only=True)
        if (checkpoint.get("preprocess") != PREPROCESS
                or checkpoint.get("class_names") != ["tilt_le_5", "tilt_gt_5"]):
            raise ValueError("Несовместимые веса spine_model; используйте результат train.py")
        self.threshold = float(checkpoint["threshold"])
        if checkpoint.get("architecture") == "spine_feature_ensemble":
            self.spine = FeaturePredictor(checkpoint, self.device)
        elif checkpoint.get("architecture") == "resnet18_gray_2":
            # Старые веса продолжают работать до повторного запуска train.py.
            model = build_model().to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            self.spine = model.eval()
        else:
            raise ValueError("Неизвестная архитектура spine_model")

    @torch.inference_mode()
    def predict(self, path):
        started = time.perf_counter()
        body_x = preprocess_body(path).to(self.device)
        probabilities = self.first(body_x).softmax(1)[0].cpu().tolist()
        body_class = CLASS_NAMES[max(range(len(probabilities)), key=probabilities.__getitem__)]
        is_spine = body_class == "spine"
        result = {
            "path": str(path), "anatomical_region": "spine" if is_spine else "hip",
            "spine_score": probabilities[CLASS_NAMES.index("spine")], "quality_class": None,
            "tilt_score": None, "threshold": None, "violation_type": None,
            "processing_status": "Success",
        }
        if is_spine:
            image = read_image(path)
            if self.spine is None:
                self._load_spine()
            if isinstance(self.spine, FeaturePredictor):
                score = self.spine.predict(image)
            else:
                x = image_tensor(image).unsqueeze(0).to(self.device)
                score = float(self.spine((x - 0.5) / 0.5).softmax(1)[0, 1].item())
            defect = int(score >= self.threshold)
            result.update(quality_class=defect, tilt_score=score, threshold=self.threshold,
                          violation_type="spine_tilt_gt_5" if defect else "none")
        else:
            result["message"] = "Наклон не оценивался: body_part_model определила бедро"
        result["time_of_processing"] = round(time.perf_counter() - started, 4)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--first-weights", type=Path, default=DEFAULT_FIRST_WEIGHTS)
    parser.add_argument("--spine-weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if not args.target.exists():
        parser.error(f"Не найдено: {args.target}")
    files = [args.target] if args.target.is_file() else sorted(
        p for p in args.target.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"
    )
    if not files:
        parser.error("DICOM-файлов не найдено")
    try:
        pipeline = SpinePipeline(args.first_weights, args.spine_weights, args.device)
    except Exception as exc:
        parser.exit(1, f"Ошибка загрузки body_part_model: {exc}\n")
    failures = 0
    for path in files:
        try:
            result = pipeline.predict(path)
        except Exception as exc:
            failures += 1
            print(f"{path}: ошибка: {exc}", file=sys.stderr)
            continue
        if result["quality_class"] is None:
            print(f"{path}: не позвоночник, наклон не оценивался", file=sys.stderr)
            continue
        print("Да" if result["quality_class"] == 1 else "Нет")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
