FROM python:3.11-slim
# Install security updates to reduce vulnerabilities

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libxcb1 libx11-6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt
COPY . . 

ENV TRANSFORMERS_NO_TF=1
ENV OMP_NUM_THREADS=1
ENV PORT=8080

# Important: Listen on $PORT
CMD sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT}}' 