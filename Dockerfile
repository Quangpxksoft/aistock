FROM python:3.10-slim

# Tắt buffering log (giúp Render log realtime)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài dependencies hệ thống (cần cho numpy/pandas/bcrypt)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Streamlit port
EXPOSE 8501

# Start app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]