# --------------------------------------------------
# Base image: lightweight Python 3.10
# --------------------------------------------------
FROM python:3.10-slim

# --------------------------------------------------
# Environment configuration
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=${PORT} \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# --------------------------------------------------
# System dependencies
# --------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Set working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Install Python dependencies first for caching
# --------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# --------------------------------------------------
# Copy all project files
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Expose the port Streamlit will run on
# --------------------------------------------------
EXPOSE ${PORT}

# --------------------------------------------------
# Run Streamlit directly (no need for start.sh)
# --------------------------------------------------
CMD ["streamlit", "run", "app.py", "--server.port=${PORT}", "--server.address=0.0.0.0"]
