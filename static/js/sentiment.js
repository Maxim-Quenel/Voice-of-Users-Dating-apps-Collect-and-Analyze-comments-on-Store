(() => {
  const dataEl = document.getElementById("sentiment-data");
  let data = { sentiment_counts: [], confidence_bins: [] };

  if (dataEl && dataEl.textContent) {
    try {
      data = JSON.parse(dataEl.textContent);
    } catch (err) {
      console.warn("Failed to parse sentiment payload.", err);
    }
  }

  const overlay = document.getElementById("loading-overlay");
  const progressBar = document.getElementById("progress-bar");
  let progressTimer;

  function startLoading() {
    if (!overlay || !progressBar) return;
    overlay.classList.remove("hidden");
    let pct = 8;
    progressBar.style.width = pct + "%";
    clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      pct = Math.min(97, pct + Math.random() * 7);
      progressBar.style.width = pct + "%";
    }, 380);
  }

  document.querySelectorAll("form").forEach(form => {
    form.addEventListener("submit", () => {
      startLoading();
    });
  });

  function ensurePlotly(cb) {
    if (window.Plotly) return cb();
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.27.0.min.js";
    s.onload = cb;
    document.head.appendChild(s);
  }

  const baseLayout = {
    font: { color: "#e2e8f0", size: 12 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    hoverlabel: { bgcolor: "#0b1220", bordercolor: "#22c55e", font: { color: "#e2e8f0" } },
    margin: { l: 50, r: 10, t: 10, b: 50 },
  };
  const baseConfig = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["toggleSpikelines", "select2d", "lasso2d"],
  };

  function plotSentimentCounts() {
    const container = document.getElementById("sentiment-counts");
    if (!container) return;
    const counts = data.sentiment_counts || [];
    if (counts.length === 0) return;

    const labels = counts.map(row => row.label);
    const values = counts.map(row => row.count);
    const colors = counts.map(row => {
      if (row.sentiment === -1) return "#f87171";
      if (row.sentiment === 1) return "#22c55e";
      return "#fbbf24";
    });
    const trace = { x: labels, y: values, type: "bar", marker: { color: colors } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotConfidenceBins() {
    const container = document.getElementById("confidence-bins");
    if (!container) return;
    const bins = data.confidence_bins || [];
    if (bins.length === 0) return;
    const labels = bins.map(row => row.bucket);
    const values = bins.map(row => row.count);
    const trace = { x: labels, y: values, type: "bar", marker: { color: "#38bdf8" } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  ensurePlotly(() => {
    plotSentimentCounts();
    plotConfidenceBins();
  });
})();
