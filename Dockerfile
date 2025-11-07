# --------------------------------------------------
# Base image
# --------------------------------------------------
FROM python:3.10-slim

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --------------------------------------------------
# System dependencies
# --------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Install dependencies
# --------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# --------------------------------------------------
# Copy all source code
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Expose Cloud Run port
# --------------------------------------------------
EXPOSE 8080

# --------------------------------------------------
# Command: bind Streamlit to Cloud Run’s PORT
# --------------------------------------------------
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
