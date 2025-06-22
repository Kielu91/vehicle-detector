# Dockerfile dla prawdziwej AI (torch + ultralytics)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Minimalne zależności systemowe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Zainstaluj Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pobierz pretrenowany model YOLO
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Skopiuj kod aplikacji
COPY . .

# Uruchom setup modelu
RUN python fix_model.py

EXPOSE 8000

WORKDIR /app/app
CMD ["python", "main.py"]