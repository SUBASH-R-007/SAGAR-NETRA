# SAGAR-NETRA — multi-stage build: React dashboard, then Python service.

# ---- Stage 1: dashboard -----------------------------------------------------
FROM node:22-slim AS web
WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci || npm install
COPY web/ ./
RUN npm run build

# ---- Stage 2: API + pipeline ------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

# OpenCV headless needs libgl-less build; libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY sonar_core/ sonar_core/
COPY tridentnet/ tridentnet/
COPY physicheck/ physicheck/
COPY geoscribe/ geoscribe/
COPY api/ api/
COPY edge/ edge/
COPY configs/ configs/
COPY scripts/ scripts/
COPY data/layers/ data/layers/

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -e .[api,ml,geo]

COPY --from=web /app/web/dist web/dist

# Weights are mounted or baked at build time when available.
RUN mkdir -p weights data/uploads outputs

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
