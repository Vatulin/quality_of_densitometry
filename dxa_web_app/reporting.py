"""Shared conclusion taxonomy and CSV contract (organizer's question 6)."""

SPINE = "Поясничный отдел позвоночника"
HIP = "Проксимальный отдел бедра"
PLACEMENT = "Некорректная укладка"
AXIS = "Не выравнена ось позвоночника"
FOREIGN_OBJECTS = "Присутствуют посторонние предметы"
ROI = "Некорректная область интереса"

VIOLATION_ORDER = {
    SPINE: (PLACEMENT, AXIS, FOREIGN_OBJECTS),
    HIP: (PLACEMENT, ROI),
}
VIOLATION_MAPPING = {
    SPINE: {
        PLACEMENT: PLACEMENT,
        "Некорректная укладка позвоночника": PLACEMENT,
        AXIS: AXIS,
        FOREIGN_OBJECTS: FOREIGN_OBJECTS,
        "Наличие артефактов": FOREIGN_OBJECTS,
    },
    HIP: {
        PLACEMENT: PLACEMENT,
        "Не выравнена ось / ротация": PLACEMENT,
        ROI: ROI,
        "Недостаточный верхний отступ ROI (менее 30 мм)": ROI,
        "Недостаточный нижний отступ ROI (менее 30 мм)": ROI,
        "Недостаточный боковой отступ ROI (менее 20 мм)": ROI,
        "Область интереса неполна у нижней границы изображения": ROI,
        "Возможен дефект ROI: верхний отступ около порога 30 мм": ROI,
        "Возможен дефект ROI: нижний отступ около порога 30 мм": ROI,
        "Возможен дефект ROI: боковой отступ около порога 20 мм": ROI,
    },
}


def canonical_violations(region: str, violations: list[str]) -> list[str]:
    """Normalize model labels before publishing a conclusion in any format."""
    labels = set()
    mapping = VIOLATION_MAPPING[region]
    for label in violations:
        # Hip artifacts remain in model_predictions, outside the conclusion.
        if region == HIP and label in {"Наличие артефактов", FOREIGN_OBJECTS}:
            continue
        # Unknown labels must fail analysis rather than produce a normal result.
        labels.add(mapping[label])
    return [label for label in VIOLATION_ORDER[region] if label in labels]


def csv_result_row(result: dict) -> dict:
    """Serialize the same conclusion exposed by the API, including manual review."""
    return {
        "path_to_study": result["path_to_study"],
        "study_uid": result["study_uid"],
        "image_uid": result["image_uid"],
        "anatomical_region": result["anatomical_region"] or "",
        "quality_class": result["quality_class"],
        "violation_type": ";".join(result["violation_type"]),
        "processing_status": result["processing_status"],
        "time_of_processing": result["time_of_processing"],
    }
