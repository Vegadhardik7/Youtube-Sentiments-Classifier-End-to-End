document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");

  // 🔐 Backend API (EC2)
  const API_URL = "http://3.87.125.175:8080";

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

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const match = url.match(/v=([a-zA-Z0-9_-]{11})/);

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
      const comments = await fetchComments(videoId);

      if (!comments.length) {
        showError("No Comments", "No comments found for this video.");
        return;
      }

      const predictions = await getSentimentPredictions(comments);

      if (!predictions) {
        showError("Prediction Failed", "Unable to analyze sentiment.");
        return;
      }

      // (rest of your logic stays EXACTLY the same)
    } catch (err) {
      console.error(err);
      showError("Error", "Backend not reachable.");
    }
  });

  async function fetchComments(videoId) {
    const res = await fetch(`${API_URL}/fetch_comments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId })
    });

    if (!res.ok) throw new Error("Failed to fetch comments");
    return res.json();
  }

  async function getSentimentPredictions(comments) {
    const res = await fetch(`${API_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments })
    });

    if (!res.ok) return null;
    return res.json();
  }

  async function fetchAndRenderImage(endpoint, payload, containerId) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Image generation failed");

    const blob = await res.blob();
    const img = document.createElement("img");
    img.src = URL.createObjectURL(blob);
    img.style.width = "100%";

    document.getElementById(containerId).appendChild(img);
  }
});
