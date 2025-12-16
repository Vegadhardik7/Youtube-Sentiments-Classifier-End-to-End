document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const loadingSpinner = document.getElementById("loading-spinner");

  // 🔐 Backend only (no secrets in frontend)
  const API_URL = "http://192.168.0.133:5000";

  function showMessage(html) {
    outputDiv.innerHTML = html;
  }

  function showError(title, msg) {
    showMessage(`
      <div class="section">
        <div class="section-title">${title}</div>
        <p>${msg}</p>
      </div>
    `);
  }

  // --------------------------------------------------
  // Get current tab URL
  // --------------------------------------------------
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex =
      /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;

    const match = url.match(youtubeRegex);

    if (!match) {
      showError("Invalid Page", "This is not a valid YouTube video URL.");
      return;
    }

    const videoId = match[1];
    showMessage(`
      <div class="section-title">YouTube Video ID</div>
      <p>${videoId}</p>
      <p>Fetching comments…</p>
    `);

    try {
      // --------------------------------------------------
      // 1. Fetch comments from backend
      // --------------------------------------------------
      const comments = await fetchComments(videoId);

      if (!comments.length) {
        showError("No Comments", "No comments found for this video.");
        return;
      }

      showMessage(`
        <p>Fetched ${comments.length} comments.</p>
        <p>Running sentiment analysis…</p>
      `);

      // --------------------------------------------------
      // 2. Sentiment prediction
      // --------------------------------------------------
      const predictions = await getSentimentPredictions(comments);

      if (!predictions) {
        showError("Prediction Failed", "Unable to analyze sentiment.");
        return;
      }

      // --------------------------------------------------
      // 3. Metrics calculation
      // --------------------------------------------------
      const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
      const sentimentTrend = [];

      let totalSentiment = 0;

      predictions.forEach((p) => {
        sentimentCounts[p.sentiment]++;
        sentimentTrend.push({
          timestamp: p.timestamp,
          sentiment: parseInt(p.sentiment),
        });
        totalSentiment += parseInt(p.sentiment);
      });

      const totalComments = comments.length;
      const uniqueUsers = new Set(comments.map((c) => c.authorId)).size;
      const totalWords = comments.reduce(
        (sum, c) => sum + c.text.split(/\s+/).length,
        0
      );

      const avgWords = (totalWords / totalComments).toFixed(2);
      const avgSentiment = (totalSentiment / totalComments).toFixed(2);
      const normalizedScore = (((+avgSentiment + 1) / 2) * 10).toFixed(2);

      // --------------------------------------------------
      // 4. Summary UI
      // --------------------------------------------------
      outputDiv.innerHTML = `
        <div class="section">
          <div class="section-title">Comment Analysis Summary</div>
          <div class="metrics-container">
            <div class="metric"><b>Total Comments</b><br>${totalComments}</div>
            <div class="metric"><b>Unique Commenters</b><br>${uniqueUsers}</div>
            <div class="metric"><b>Avg Comment Length</b><br>${avgWords} words</div>
            <div class="metric"><b>Avg Sentiment</b><br>${normalizedScore}/10</div>
          </div>
        </div>

        <div class="section">
          <div class="section-title">Sentiment Distribution</div>
          <div id="chart-container"></div>
        </div>

        <div class="section">
          <div class="section-title">Sentiment Trend</div>
          <div id="trend-graph-container"></div>
        </div>

        <div class="section">
          <div class="section-title">Comment Wordcloud</div>
          <div id="wordcloud-container"></div>
        </div>

        <div class="section">
          <div class="section-title">Top 25 Comments</div>
          <ul class="comment-list">
            ${predictions.slice(0, 25).map((p, i) => `
              <li>
                <b>${i + 1}.</b> ${p.comment}<br>
                <small>Sentiment: ${p.sentiment}</small>
              </li>
            `).join("")}
          </ul>
        </div>
      `;

      // --------------------------------------------------
      // 5. Charts
      // --------------------------------------------------
      await fetchAndRenderImage(
        "/generate_chart",
        { sentiment_counts: sentimentCounts },
        "chart-container"
      );

      await fetchAndRenderImage(
        "/generate_trend_graph",
        { sentiment_data: sentimentTrend },
        "trend-graph-container"
      );

      await fetchAndRenderImage(
        "/generate_wordcloud",
        { comments: comments.map((c) => c.text) },
        "wordcloud-container"
      );
    } catch (err) {
      console.error(err);
      showError("Error", "Something went wrong. Check backend logs.");
    }
  });

  // --------------------------------------------------
  // API helpers
  // --------------------------------------------------
  async function fetchComments(videoId) {
    const res = await fetch(`${API_URL}/fetch_comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
    });

    if (!res.ok) throw new Error("Failed to fetch comments");
    return res.json();
  }

  async function getSentimentPredictions(comments) {
    const res = await fetch(`${API_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });

    if (!res.ok) return null;
    return res.json();
  }

  async function fetchAndRenderImage(endpoint, payload, containerId) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error("Image generation failed");

    const blob = await res.blob();
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    img.style.width = "100%";

    document.getElementById(containerId).appendChild(img);
  }
});
