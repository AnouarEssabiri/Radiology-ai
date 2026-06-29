FROM python:3.11-slim

WORKDIR /app

# Install git, git-lfs, and build essentials
RUN apt-get update && apt-get install -y git git-lfs build-essential && rm -rf /var/lib/apt/lists/*

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
