# Use Python 3.11 as it has the best "pre-compiled" support for your pinned libraries

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	ffmpeg \
	libsndfile1 \
	&& rm -rf /var/lib/apt/lists/\*

WORKDIR /app

# Upgrade pip to ensure it looks for the latest wheels

RUN pip install --upgrade pip

COPY requirements.txt .

# Use --prefer-binary to force pip to use pre-compiled versions

# instead of trying to build from source

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT