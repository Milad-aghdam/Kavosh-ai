FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache


WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

RUN mkdir -p /app/cache && chown -R user:user /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY --chown=user:user . .


USER user


EXPOSE 7860

CMD ["python", "app.py"]