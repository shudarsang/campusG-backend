FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 \
    EMBEDDING_CACHE_DIR=/app/models

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. models/ is gitignored (it
# is a 65 MB build artefact), so without this step the container
# downloads it on first request - on every cold start, while a user
# is waiting for an answer.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

# Copy application code and knowledge base
COPY . .

# Expose port (7860 is default for Hugging Face Spaces)
EXPOSE 7860

# Run uvicorn (reads $PORT dynamically or defaults to 7860)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
