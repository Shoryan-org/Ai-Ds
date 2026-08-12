# Use a slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for sentence-transformers / faiss
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including Knowledge/, vector_db/, scripts/, generation/, fastapi_app/)
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Default command: run the FastAPI server
# (The Gradio UI is optional and can be run separately if needed)
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]