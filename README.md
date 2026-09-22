зависимости при необходимости:
```
pip install fastapi uvicorn python-multipart pandas pydicom torch torchvision jinja2 pillow
```

запуск сервера из папки `dxa_web_app`:
```
uvicorn main:app --reload
```

В интерфейсе можно выбрать или перетащить несколько `.dcm`, `.dicom` и `.zip`
одновременно. Они обрабатываются одной задачей с общим отчётом CSV.
Файлы с одинаковыми именами сохраняются в отдельных каталогах.

API: `POST /api/analyze` принимает несколько multipart-полей `files`.
Прежнее поле `file` для одного файла также поддерживается.
Ответ содержит `task_id`; статус и результаты доступны через
`GET /api/status/{task_id}`, отчёт — `GET /api/download/{task_id}`.

CSV в UTF-8 с BOM содержит колонки из п. 2.5 ТЗ в указанном порядке:
`path_to_study`, `study_uid`, `image_uid`, `anatomical_region`, `quality_class`,
`violation_type`, `processing_status`, `time_of_processing`.
Дополнительная колонка `error_message` содержит причину технической ошибки.
Путь указывает на сохранённый файл на сервере, UID берутся из DICOM-тегов
StudyInstanceUID и SOPInstanceUID (при отсутствии остаются пустыми).
Время измеряется в секундах для каждого изображения, включая чтение его метаданных
и инференс, без времени загрузки и распаковки архива.
Класс качества — целое `0` или `1`; при `Failure` класс остаётся пустым.
Сбой отдельного файла не останавливает пакет. Повреждённый ZIP или ZIP без
DICOM-файлов отражается отдельной строкой `Failure` с путём к архиву.
