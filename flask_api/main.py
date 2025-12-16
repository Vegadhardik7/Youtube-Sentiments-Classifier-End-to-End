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
from dotenv import load_dotenv
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# ENV SETUP
# ------------------------------------------------------------------
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://ec2-13-221-127-40.compute-1.amazonaws.com:5000"
)

if not YOUTUBE_API_KEY:
    raise RuntimeError("❌ YOUTUBE_API_KEY not found in environment")

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask-api")

# ------------------------------------------------------------------
# APP INIT
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
def load_model_and_vectorizer():
    logger.info("🔄 Loading ML model from MLflow...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = "models:/yt_chrome_plugin_model@production"
    model = mlflow.pyfunc.load_model(model_uri)

    with open("./tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    logger.info("✅ Model and vectorizer loaded successfully")
    return model, vectorizer


model, vectorizer = load_model_and_vectorizer()

# ------------------------------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - {
            "not", "no", "but", "however"
        }

        words = [w for w in comment.split() if w not in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]

        return " ".join(words)
    except Exception:
        return comment

# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------
@app.route("/")
def home():
    return "Flask ML API is running 🚀"

# ------------------------------------------------------------------
# YOUTUBE COMMENTS FETCH
# ------------------------------------------------------------------
@app.route("/fetch_comments", methods=["POST"])
def fetch_comments():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    comments = []
    page_token = ""

    try:
        while len(comments) < 500:
            url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": 100,
                "pageToken": page_token,
                "key": YOUTUBE_API_KEY,
            }

            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            res = r.json()

            for item in res.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": snippet["textOriginal"],
                    "timestamp": snippet["publishedAt"]
                })

            page_token = res.get("nextPageToken")
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
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [c["text"] for c in comments_data]
        timestamps = [c["timestamp"] for c in comments_data]

        # Preprocess
        processed = [preprocess_comment(c) for c in comments]

        # Vectorize
        X = vectorizer.transform(processed)

        # ✅ Convert to DataFrame with correct feature names
        feature_names = vectorizer.get_feature_names_out()
        X_df = pd.DataFrame(X.toarray(), columns=feature_names)

        # Predict via MLflow
        preds = model.predict(X_df)
        preds = preds.astype(int).tolist()

        return jsonify([
            {
                "comment": c,
                "sentiment": str(p),
                "timestamp": t
            }
            for c, p, t in zip(comments, preds, timestamps)
        ])

    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500


# ------------------------------------------------------------------
# PIE CHART
# ------------------------------------------------------------------
@app.route("/generate_chart", methods=["POST"])
def generate_chart():
    data = request.get_json()
    counts = data.get("sentiment_counts")

    sizes = [
        int(counts.get("1", 0)),
        int(counts.get("0", 0)),
        int(counts.get("-1", 0)),
    ]

    if sum(sizes) == 0:
        return jsonify({"error": "No data"}), 400

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.axis("equal")

    img = io.BytesIO()
    plt.savefig(img, format="PNG", transparent=True)
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")

# ------------------------------------------------------------------
# WORD CLOUD
# ------------------------------------------------------------------
@app.route("/generate_wordcloud", methods=["POST"])
def generate_wordcloud():
    data = request.get_json()
    comments = data.get("comments")

    text = " ".join(preprocess_comment(c) for c in comments)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=set(stopwords.words("english")),
    ).generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)

    return send_file(img, mimetype="image/png")

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Flask server on port 8080")
    app.run(host="0.0.0.0", port=8080)