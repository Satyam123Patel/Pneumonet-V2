FROM python:3.10-slim-bullseye

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Copy dependency definition and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Expose Hugging Face's default port
EXPOSE 7860

# Run Flask application using Gunicorn for production-grade stability
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]