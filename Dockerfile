FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install the project's pinned dependencies before copying large model weights.
COPY requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt && python -m pip check

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/uploads \
    && chown app:app /app/uploads

COPY dxa_web_app/ ./dxa_web_app/
COPY models/ ./models/

USER app

# Fail the build if a checkpoint cannot be loaded; startup otherwise only logs it.
RUN python -c "from pathlib import Path; from dxa_web_app.main import MultiModelAnalyzer; a = MultiModelAnalyzer(Path('/app/models')); assert len(a.models) == 6, a.model_load_errors"

EXPOSE 8000
VOLUME ["/app/uploads"]

# Liveness check; model compatibility is checked during the build above.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).close()"

# Tasks are stored in process memory, so the API must use a single worker.
CMD ["python", "-m", "uvicorn", "dxa_web_app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
