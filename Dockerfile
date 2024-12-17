# Base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the application port
EXPOSE 5000

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Run the script
CMD ["python", "test.py"]
