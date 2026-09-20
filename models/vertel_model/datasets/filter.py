import numpy as np
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("dataset_for_first_model.xlsx")
LABELS_PATH  = Path("разметка.xlsx")

df = pd.read_excel(DATASET_PATH)
df["study_id"] = df["study_id"].astype(str).str.strip()

hip_df = df[(df["body_hip_right"] == 1) | (df["body_hip_left"] == 1)].copy()
print(f"[1] Всего снимков: {len(df)}, из них бедро: {len(hip_df)}")

raw  = pd.read_excel(LABELS_PATH, sheet_name="Калибровка", header=None)
data = raw.iloc[2:].copy()
data = data[data[1].notna()]

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

labels = pd.DataFrame({
    "study":                          data[1].astype(str).str.strip(),
    "right_hip_positioning_rotation": to_num(data[5]),
    "right_hip_roi":                  to_num(data[6]),
    "left_hip_positioning_rotation":  to_num(data[7]),
    "left_hip_roi":                   to_num(data[8]),
}).drop_duplicates("study")

print(f"[2] Размеченных исследований: {len(labels)}")


right = hip_df[hip_df["body_hip_right"] == 1].copy(); right["side"] = "right"
left  = hip_df[hip_df["body_hip_left"]  == 1].copy(); left["side"]  = "left"
hip_long = pd.concat([right, left], ignore_index=True)
hip_long = hip_long.merge(labels, left_on="study_id", right_on="study", how="left")

hip_long["positioning_rotation"] = np.where(
    hip_long["side"] == "right",
    hip_long["right_hip_positioning_rotation"],
    hip_long["left_hip_positioning_rotation"],
)
hip_long["roi_correctness"] = np.where(
    hip_long["side"] == "right",
    hip_long["right_hip_roi"],
    hip_long["left_hip_roi"],
)
hip_long["positioning_rotation"] = hip_long["positioning_rotation"].astype("float")
hip_long["roi_correctness"]      = hip_long["roi_correctness"].astype("float")


hip_long = hip_long.drop(columns=[
    "study", "body_spine", "body_hip_right", "body_hip_left",
    "right_hip_positioning_rotation", "right_hip_roi",
    "left_hip_positioning_rotation",  "left_hip_roi",
])

unmatched = sorted(set(hip_long["study_id"]) - set(labels["study"]))
if unmatched:
    print(f"[!] {len(unmatched)} исследований бедра без разметки:")
    for s in unmatched[:10]:
        print(f"     {s}")

for side in ["right", "left"]:
    sub = hip_long[hip_long["side"] == side]
    lab = sub[["positioning_rotation", "roi_correctness"]].notna().any(axis=1).sum()
    print(f"  {side:5s}: n={len(sub):4d}, labeled={lab:4d}, "
          f"pos_rot={sub['positioning_rotation'].value_counts(dropna=False).to_dict()}, "
          f"roi={sub['roi_correctness'].value_counts(dropna=False).to_dict()}")

hip_long.to_csv("dataset_for_hip_model.csv", index=False)
hip_long.to_excel("dataset_for_hip_model.xlsx", index=False)
