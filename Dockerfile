# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy application code ----------
COPY flask_api ./flask_api
COPY src ./src

# ---------- Copy ML artifacts ----------
COPY lgbm_model.pkl .
COPY tfidf_vectorizer.pkl .

# ---------- Expose Flask port ----------
EXPOSE 5000

# ---------- Run Flask app ----------
CMD ["python3", "flask_api/main.py"]