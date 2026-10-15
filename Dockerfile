FROM python:3.12-slim

WORKDIR /app

# Dépendances système pour Box2D et les vidéos
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement le code et le modèle
COPY src/ ./src/
COPY models/EAGLE1_TOP_MODEL_PPO_16ENV.zip ./models/EAGLE1_TOP_MODEL_PPO_16ENV.zip

# Dossier des vidéos
RUN mkdir -p outputs/videos

# Port utilisé par Render
ENV PORT=10000
EXPOSE 10000

# Démarrer uniquement FastAPI
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]