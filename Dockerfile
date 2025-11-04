# Use a slim Python base
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Streamlit defaults
ENV PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=${PORT}
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# system deps (git used for huggingface downloads, build-essential for any wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install wheel then the python deps
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of the repo
COPY . .

# Make start script executable
RUN chmod +x /app/start.sh

# Expose Streamlit port
EXPOSE ${PORT}

# Start: download fallback model if required, then launch streamlit
CMD ["/app/start.sh"]
