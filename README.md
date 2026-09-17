зависимости при необходимости:
```
pip install fastapi uvicorn python-multipart pandas pydicom torch torchvision jinja2 pillow
```

запуск сервера из папки `dxa_web_app`:
```
uvicorn main:app --reload
```