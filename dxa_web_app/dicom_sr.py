"""Basic Text SR export of the current DXA quality conclusion.

Local concepts use the private 99DXA coding scheme; no standard template is claimed.
Reports are UNVERIFIED: editing a conclusion is not a signed clinical attestation.
"""
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
from zipfile import ZipFile, ZIP_DEFLATED

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import BasicTextSRStorage, ExplicitVRLittleEndian, UID, generate_uid
from pydicom.valuerep import validate_value
from fastapi import HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)


def concept(value, meaning):
    code = Dataset()
    code.CodeValue = value
    code.CodingSchemeDesignator = "99DXA"
    code.CodeMeaning = meaning
    return [code]


def text_item(code, meaning, text):
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = concept(code, meaning)
    item.TextValue = text
    return item


def build_sr(source, result):
    """Create a new report without modifying the source dataset or result."""
    if result.get("processing_status") != "Success" or result.get("quality_class") not in (0, 1):
        raise ValueError("DICOM SR недоступен: анализ изображения не завершён успешно")
    labels = result.get("violation_type", [])
    if (not isinstance(labels, list) or any(not isinstance(x, str) or not x.strip() for x in labels)
            or bool(labels) != bool(result["quality_class"])):
        raise ValueError("Заключение содержит несогласованные класс качества и нарушения")
    identifiers = {name: str(getattr(source, name, "") or "") for name in
                   ("StudyInstanceUID", "SeriesInstanceUID", "SOPClassUID", "SOPInstanceUID")}
    # Read invalid input tolerantly, but never write invalid UIDs or invented
    # image references to the report. In particular, do not strip leading zeros:
    # a PACS may match the original identifier as a literal string.
    invalid = [name for name, value in identifiers.items()
               if not value or not UID(value, validation_mode=pydicom.config.IGNORE).is_valid]
    study_uid = identifiers["StudyInstanceUID"]
    if "StudyInstanceUID" in invalid:
        # Repeat exports share the fallback study. Missing studies are scoped
        # to the source file; a batch may contain unrelated patients/studies.
        identity = study_uid or str(result.get("path") or source.to_json())
        study_uid = generate_uid(entropy_srcs=["DXA-SR-fallback-study-v1", identity])

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = BasicTextSRStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"
    invalid_metadata = []

    def copy_metadata(name, required=False):
        element = source.data_element(name) if name in source else None
        value = deepcopy(element.value) if element is not None else ""
        if element is not None:
            try:
                values = value if element.VM > 1 else [value]
                for item in values:
                    validate_value(element.VR, str(item), pydicom.config.RAISE)
            except (ValueError, TypeError):
                invalid_metadata.append(name)
                if not required:
                    return
                value = ""
        if required or element is not None:
            setattr(ds, name, value)

    # Type 2 attributes remain present even in anonymized input.
    for name in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                 "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyID", "AccessionNumber"):
        copy_metadata(name, required=True)
    for name in ("IssuerOfPatientID", "StudyDescription", "PatientIdentityRemoved", "DeidentificationMethod"):
        copy_metadata(name)
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "SR"
    ds.SeriesNumber = 900
    ds.InstanceNumber = 1
    ds.SeriesDescription = "DXA quality report"
    ds.Manufacturer = "DXA Multi-Model Analyzer"
    ds.ReferencedPerformedProcedureStepSequence = []
    ds.PerformedProcedureCodeSequence = []
    now = datetime.now(timezone.utc)
    ds.ContentDate = ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.ContentTime = ds.InstanceCreationTime = now.strftime("%H%M%S.%f")
    ds.TimezoneOffsetFromUTC = "+0000"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"
    ds.PreliminaryFlag = "PRELIMINARY"
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = concept("QUALITY_REPORT", "DXA quality report")

    items = []
    if not invalid:
        reference = Dataset()
        reference.ReferencedSOPClassUID = source.SOPClassUID
        reference.ReferencedSOPInstanceUID = source.SOPInstanceUID
        series = Dataset()
        series.SeriesInstanceUID = source.SeriesInstanceUID
        series.ReferencedSOPSequence = [reference]
        study = Dataset()
        study.StudyInstanceUID = source.StudyInstanceUID
        study.ReferencedSeriesSequence = [series]
        ds.CurrentRequestedProcedureEvidenceSequence = [study]
        image = Dataset()
        image.RelationshipType = "CONTAINS"
        image.ValueType = "IMAGE"
        image.ConceptNameCodeSequence = concept("SOURCE", "Source image")
        image.ReferencedSOPSequence = [deepcopy(reference)]
        items.append(image)
    else:
        items.append(text_item("SOURCE_FILE", "Source file",
                               str(result.get("filename") or "Имя файла не указано")))
        items.append(text_item("SOURCE_UIDS", "Original source identifiers",
                               "\n".join(f"{name}: {value or 'отсутствует'}"
                                         for name, value in identifiers.items())))
        note = ("Автоматическая привязка к исходному изображению недоступна: "
                "отсутствуют или некорректны идентификаторы " + ", ".join(invalid) + ".")
        if "StudyInstanceUID" in invalid:
            note += " Для SR сформирован новый идентификатор исследования."
        items.append(text_item("SOURCE_WARNING", "Source metadata warning", note))
    region = result.get("anatomical_region") or "Не определена"
    side = {"right": "Правая", "left": "Левая"}.get(result.get("hip_side"))
    items.append(text_item("REGION", "Anatomical region", region))
    if invalid_metadata:
        items.append(text_item("METADATA_WARNING", "Invalid source metadata",
                               "Некорректные значения исходных метаданных не перенесены в SR: "
                               + ", ".join(invalid_metadata) + "."))
    if side:
        items.append(text_item("SIDE", "Side", side))
    items.append(text_item("CONCLUSION", "Quality conclusion",
                           "Выявлены нарушения качества исследования." if labels else
                           "По результатам анализа нарушения качества не выявлены."))
    items.extend(text_item("VIOLATION", "Quality violation", label) for label in labels)
    items.append(text_item("METHOD", "Conclusion source", "Автоматический анализ качества DXA."))
    if result.get("model_conclusion"):
        original = result["model_conclusion"]["violation_type"]
        items.append(text_item("MODEL_RESULT", "Original model conclusion",
                               "; ".join(original) or "Нарушения не выявлены."))
        items.append(text_item("REVIEW", "Manual review",
                               "Заключение просмотрено вручную. Дата: " + result.get("reviewed_at", "не указана")))
    ds.ContentSequence = items
    return ds


def report_bytes(result):
    if result.get("processing_status") != "Success":
        raise ValueError(result.get("error_message") or "Анализ изображения не завершён успешно")
    source = pydicom.dcmread(result["path"], stop_before_pixels=True)
    report = build_sr(source, result)
    stream = BytesIO()
    report.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def register_dicom_sr(app, tasks):
    def completed_task(task_id):
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Задача не найдена")
        if task.get("status") != "COMPLETED":
            raise HTTPException(409, "Дождитесь завершения анализа")
        return task

    @app.get("/api/dicom-sr/{task_id}/{index}")
    def download_sr(task_id: str, index: int):
        task = completed_task(task_id)
        if index < 0 or index >= len(task["results"]):
            raise HTTPException(404, "Изображение не найдено")
        try:
            data = report_bytes(deepcopy(task["results"][index]))
        except Exception as exc:
            logger.exception("DICOM SR export failed")
            raise HTTPException(422, f"Не удалось сформировать DICOM SR: {exc}") from exc
        return Response(data, media_type="application/dicom", headers={
            "Content-Disposition": f'attachment; filename="report_{index + 1}.dcm"'})

    @app.get("/api/dicom-sr/{task_id}")
    def download_sr_archive(task_id: str):
        results = deepcopy(completed_task(task_id)["results"])
        stream = BytesIO()
        manifest = []
        with ZipFile(stream, "w", ZIP_DEFLATED) as archive:
            for index, result in enumerate(results):
                entry = {"index": index, "filename": result.get("filename", "")}
                try:
                    data = report_bytes(result)
                    name = f"report_{index + 1}.dcm"
                    archive.writestr(name, data)
                    entry.update(status="Success", report=name)
                except Exception as exc:
                    entry.update(status="Failure", error=str(exc))
                manifest.append(entry)
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return Response(stream.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": 'attachment; filename="dicom_sr_reports.zip"'})
