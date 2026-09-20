import sys, hashlib
import pandas as pd
import pydicom

from pathlib import Path

DATASET_PATH = Path("НД_для_обучения/Исследования")


if not DATASET_PATH.exists():
    sys.exit("Папка не найдена")

def try_read(p):
    try:
        ds = pydicom.dcmread(str(p), force=True)
        if "PixelData" not in ds:
            return None
        arr = ds.pixel_array
        if arr.size == 0:
            return None
        return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception:
        return None


dcm_files = list(DATASET_PATH.rglob("*.dcm"))

rows = []
for p in dcm_files:
    ph = try_read(p)
    if ph is None:
        continue
    rel = p.relative_to(DATASET_PATH)
    rows.append({
        "study_id": rel.parts[0],
        "series_id": rel.parts[1] if len(rel.parts) > 1 else "",
        "file_name": p.name,
        "rel_path": (DATASET_PATH / rel).as_posix(),
        "pixel_hash": ph,
    })

print(f"Прочитано DICOM: {len(rows)}")
if not rows:
    sys.exit("Ни одного валидного DICOM не найдено.")

df = pd.DataFrame(rows)

df = df.sort_values(["study_id", "series_id", "file_name"]).reset_index(drop=True)
mask_canonical = ~df.duplicated(["study_id", "pixel_hash"])
n_dups = (~mask_canonical).sum()
df = df[mask_canonical].reset_index(drop=True)


df["body_spine"]     = pd.NA   # 1 = позвоночник
df["body_hip_right"] = pd.NA   # 1 = правое бедро
df["body_hip_left"]  = pd.NA   # 1 = левое бедро

for c in ["body_spine", "body_hip_right", "body_hip_left"]:
    df[c] = df[c].astype("Int64")

df.to_excel("dataset_for_first_model.xlsx", index=False)
