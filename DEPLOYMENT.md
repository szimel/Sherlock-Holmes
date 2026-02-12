# Use a slim image to save RAM

FROM python:3.11-slim

# Install system dependencies for audio processing

# 'libsndfile1' is mandatory for the 'soundfile' library

RUN apt-get update && apt-get install -y --no-install-recommends \
 ffmpeg \
 libsndfile1 \
 && rm -rf /var/lib/apt/lists/\*

WORKDIR /app

# Install Python dependencies

# Using --no-cache-dir saves disk space and small amounts of RAM during build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code

COPY . .

# Railway provides the port via an environment variable

ENV PORT=8080

# Use a shell-format command to ensure $PORT is properly read

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
