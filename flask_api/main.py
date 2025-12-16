import os
import io
import re
import pickle
import requests
import matplotlib
matplotlib.use('Agg')
import mlflow
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
from dotenv import load_dotenv

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# ENV SETUP
# ------------------------------------------------------------------
load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("YOUTUBE_API_KEY not found in environment")

# ------------------------------------------------------------------
# APP INIT
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------
# def load_model(model_path, vectorizer_path):
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     with open(vectorizer_path, 'rb') as f:
#         vectorizer = pickle.load(f)
#     return model, vectorizer


# model, vectorizer = load_model(
#     "./lgbm_model.pkl",
#     "./tfidf_vectorizer.pkl"
# )

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://ec2-13-221-127-40.compute-1.amazonaws.com:5000/")  # Replace with your MLflow tracking URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
   
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed

# ------------------------------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {
            'not', 'but', 'however', 'no', 'yet'
        }

        words = [
            w for w in comment.split()
            if w not in stop_words
        ]

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
# 🔐 YOUTUBE COMMENTS FETCH (SECURE)
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
                    "timestamp": snippet["publishedAt"],
                    "authorId": snippet.get("authorChannelId", {}).get("value", "Unknown")
                })

            page_token = res.get("nextPageToken")
            if not page_token:
                break

        return jsonify(comments)

    except Exception as e:
        app.logger.error(f"YouTube fetch failed: {e}")
        return jsonify({"error": "Failed to fetch YouTube comments"}), 500

# ------------------------------------------------------------------
# SENTIMENT PREDICTION (WITH TIMESTAMPS)
# ------------------------------------------------------------------
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.get_json()
    comments_data = data.get("comments")

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [c["text"] for c in comments_data]
        timestamps = [c["timestamp"] for c in comments_data]

        processed = [preprocess_comment(c) for c in comments]
        X = vectorizer.transform(processed).toarray()

        preds = model.predict(X).astype(int).tolist()

        response = [
            {
                "comment": c,
                "sentiment": str(p),
                "timestamp": t
            }
            for c, p, t in zip(comments, preds, timestamps)
        ]

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# ------------------------------------------------------------------
# PIE CHART
# ------------------------------------------------------------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    data = request.get_json()
    counts = data.get("sentiment_counts")

    if not counts:
        return jsonify({"error": "No sentiment counts"}), 400

    sizes = [
        int(counts.get("1", 0)),
        int(counts.get("0", 0)),
        int(counts.get("-1", 0))
    ]

    if sum(sizes) == 0:
        return jsonify({"error": "Empty data"}), 400

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=["Positive", "Neutral", "Negative"],
        autopct="%1.1f%%",
        startangle=140
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
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    data = request.get_json()
    comments = data.get("comments")

    if not comments:
        return jsonify({"error": "No comments"}), 400

    text = " ".join(preprocess_comment(c) for c in comments)

    wc = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=set(stopwords.words("english"))
    ).generate(text)

    img = io.BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)

    return send_file(img, mimetype="image/png")

# ------------------------------------------------------------------
# TREND GRAPH
# ------------------------------------------------------------------
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    data = request.get_json()
    sentiment_data = data.get("sentiment_data")

    if not sentiment_data:
        return jsonify({"error": "No sentiment data"}), 400

    df = pd.DataFrame(sentiment_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["sentiment"] = df["sentiment"].astype(int)
    df.set_index("timestamp", inplace=True)

    monthly = (
        df.resample("M")["sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )

    perc = monthly.div(monthly.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(12, 6))
    for val, label in [(-1, "Negative"), (0, "Neutral"), (1, "Positive")]:
        if val in perc:
            plt.plot(perc.index, perc[val], marker="o", label=label)

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="PNG")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)