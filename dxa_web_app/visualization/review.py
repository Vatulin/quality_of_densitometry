"""Editable conclusions; model predictions and image annotations stay immutable."""
from datetime import datetime, timezone
from threading import Lock

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ConclusionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    violations: list[str]
    revision: int = Field(ge=0)


def register_review(app, tasks):
    lock = Lock()

    @app.put("/api/conclusion/{task_id}/{index}")
    def update_conclusion(task_id: str, index: int, change: ConclusionUpdate):
        with lock:
            task = tasks.get(task_id)
            if task is None or index < 0 or index >= len(task.get("results", [])):
                raise HTTPException(404, "Исследование не найдено")
            if task.get("status") != "COMPLETED":
                raise HTTPException(409, "Дождитесь завершения анализа")
            row = task["results"][index]
            if row.get("processing_status") != "Success":
                raise HTTPException(409, "Заключение недоступно при ошибке анализа")
            if change.revision != row.get("review_revision", 0):
                raise HTTPException(409, "Заключение уже изменено. Откройте актуальные результаты заново.")
            original = row.get("model_conclusion") or {
                "violation_type": list(row["violation_type"]),
                "quality_class": row["quality_class"],
                "quality_prob": row.get("quality_prob"),
            }
            if len(set(change.violations)) != len(change.violations) or not set(change.violations) <= set(original["violation_type"]):
                raise HTTPException(422, "Можно только отменять или восстанавливать нарушения, отмеченные моделью")
            kept = [label for label in original["violation_type"] if label in change.violations]
            edited = kept != original["violation_type"]
            updated = dict(row, model_conclusion=original, violation_type=kept,
                           quality_class=int(bool(kept)),
                           quality_prob=None if edited else original["quality_prob"],
                           manually_reviewed=edited,
                           review_revision=row.get("review_revision", 0)+1,
                           reviewed_at=datetime.now(timezone.utc).isoformat())
            task["results"][index] = updated
            return updated
