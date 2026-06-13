FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    VNSTOCK_DISABLE_UPGRADE_CHECK=1
WORKDIR /app
# System dependencies (gọn lại, đủ dùng cho numpy/pandas/bcrypt)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Install dependencies trước để cache layer
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
# Copy source code
COPY . .
EXPOSE 8501
# Run Streamlit (ổn định hơn)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
