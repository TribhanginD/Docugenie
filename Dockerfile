# 1) Base Python slim
FROM python:3.11-slim

# 2) Don’t write .pyc, unbuffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Make repo root importable
ENV PYTHONPATH=/app

# 4) Install build tools for native deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 5) Set workdir
WORKDIR /app

# 6) Copy & install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 7) Pre-warm (download) your models into HF cache
RUN python3 <<EOF
from sentence_transformers import SentenceTransformer, CrossEncoder
print("Warming embedding model…")
SentenceTransformer("all-mpnet-base-v2")
print("Warming re-ranker model…")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
EOF

# 8) Copy rest of your app
COPY . .

# 9) Expose Streamlit port
EXPOSE 8501

# 10) Launch your Welcome/Chatbot
CMD ["streamlit", "run", "welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]
