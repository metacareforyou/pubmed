FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY evidence ./evidence

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "evidence"]
