FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Skopiuj requirements.txt z głównego katalogu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Skopiuj zawartość folderu app/ do głównego katalogu kontenera (/app/)
COPY app/ ./
# Skopiuj fix_model.py z głównego katalogu
COPY fix_model.py .

RUN python fix_model.py

EXPOSE 8000
CMD ["python", "main.py"]