FROM python:3.12-slim

LABEL description="AI4Animation / MotionPredictionProject - headless training & inference container"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    libbsd0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY setup.py .
COPY ai4animation ./ai4animation

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY . .

CMD ["bash"]
