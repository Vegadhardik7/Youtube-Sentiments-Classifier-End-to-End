import os
import io
import re
import pickle
import logging
import requests
import matplotlib

matplotlib.use("Agg")

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-13-221-127-40.compute-1.amazonaws.com:5000"
)
MODEL_NAME = os.getenv("MODEL_NAME", "yt_chrome_plugin_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "./tfidf_vectorizer.pkl")

if not YOUTUBE_API_KEY:
    raise RuntimeError("❌ YOUTUBE_API_KEY not found in environment")

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask-api")

# ------------------------------------------------------------------
# FLASK INIT
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
def load_model_and_vectorizer():
    logger.info("🔄 Loading ML model from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info("📦 Loading TF-IDF vectorizer...")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info("✅ Model and vectorizer loaded successfully")
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()

# ------------------------------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------------------------------
stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no"}
lemmatizer = WordNetLemmatizer()


def preprocess_comment(text: str) -> str:
    try:
        text = text.lower()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^a-z0-9\s!?.,]", "", text)

        words = [
            lemmatizer.lemmatize(w)
            for w in text.split()
            if w not in stop_words
        ]
        return " ".join(words)
    except Exception:
        return text


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route("/")
def health():
    return "✅ YouTube Sentiment API is running"


# ------------------------------------------------------------------
# FETCH COMMENTS
# ------------------------------------------------------------------
@app.route("/fetch_comments", methods=["POST"])
def fetch_comments():
    video_id = request.json.get("video_id")
    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    comments = []
    page_token = ""

    try:
        while len(comments) < 500:
            resp = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params={
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": 100,
                    "pageToken": page_token,
                    "key": YOUTUBE_API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textOriginal"],
                    "timestamp": snippet["publishedAt"],
                })

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return jsonify(comments)

    except Exception as e:
        logger.exception("YouTube fetch failed")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# SENTIMENT PREDICTION
# ------------------------------------------------------------------
@app.route("/predict_with_timestamps", methods=["POST"])
def predict():
    data = request.json.get("comments")
    if not data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        texts = [c["text"] for c in data]
        timestamps = [c["timestamp"] for c in data]

        processed = [preprocess_comment(t) for t in texts]
        X = vectorizer.transform(processed)

        preds = model.predict(X).astype(int).tolist()

        return jsonify([
            {"comment": t, "sentiment": p, "timestamp": ts}
            for t, p, ts in zip(texts, preds, timestamps)
        ])

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# PIE CHART
# ------------------------------------------------------------------
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    counts = request.json.get("sentiment_counts")
    if not counts:
        return jsonify({"error": "No sentiment counts"}), 400

    sizes = [
        int(counts.get("1", 0)),
        int(counts.get("0", 0)),
        int(counts.get("-1", 0)),
    ]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.axis("equal")

    img = io.BytesIO()
    plt.savefig(img, format="PNG")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")


# ------------------------------------------------------------------
# WORD CLOUD
# ------------------------------------------------------------------
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    comments = request.json.get("comments")
    if not comments:
        return jsonify({"error": "No comments"}), 400

    text = " ".join(preprocess_comment(c) for c in comments)
    wc = WordCloud(width=800, height=400, background_color="black").generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)

    return send_file(img, mimetype="image/png")


# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
