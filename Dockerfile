FROM python:3.11-slim

WORKDIR /app

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Pull LFS files (model and data)
RUN git lfs pull

# Expose the port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
