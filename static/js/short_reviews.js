(() => {
  const dataEl = document.getElementById("short-data");
  let data = {
    length_counts: [],
    short_label_counts: [],
    short_word_counts: [],
    short_rating_counts: null,
    short_app_counts: null,
  };

  if (dataEl && dataEl.textContent) {
    try {
      data = JSON.parse(dataEl.textContent);
    } catch (err) {
      console.warn("Failed to parse short reviews payload.", err);
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
    hoverlabel: { bgcolor: "#0b1220", bordercolor: "#f97316", font: { color: "#e2e8f0" } },
    margin: { l: 50, r: 10, t: 10, b: 50 },
  };
  const baseConfig = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["toggleSpikelines", "select2d", "lasso2d"],
  };

  function plotLengthCounts() {
    const container = document.getElementById("length-counts");
    if (!container) return;
    const counts = data.length_counts || [];
    if (counts.length === 0) return;
    const sorted = counts.slice().sort((a, b) => a.length - b.length);
    const labels = sorted.map(row => row.length === 0 ? "Courts" : "Longs");
    const values = sorted.map(row => row.count);
    const trace = { x: labels, y: values, type: "bar", marker: { color: "#f97316" } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotShortLabelCounts() {
    const container = document.getElementById("short-label-counts");
    if (!container) return;
    const counts = data.short_label_counts || [];
    if (counts.length === 0) return;
    const labels = counts.map(row => row.label);
    const values = counts.map(row => row.count);
    const trace = { x: labels, y: values, type: "bar", marker: { color: "#38bdf8" } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotShortWordCounts() {
    const container = document.getElementById("short-word-counts");
    if (!container) return;
    const counts = data.short_word_counts || [];
    if (counts.length === 0) return;
    const labels = counts.map(row => String(row.words));
    const values = counts.map(row => row.count);
    const trace = { x: labels, y: values, type: "bar", marker: { color: "#facc15" } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotShortRatingCounts() {
    const container = document.getElementById("short-rating-counts");
    if (!container) return;
    const counts = data.short_rating_counts;
    if (!counts || counts.length === 0) return;
    const labels = counts.map(row => String(row.rating));
    const values = counts.map(row => row.count);
    const trace = { x: labels, y: values, type: "bar", marker: { color: "#22d3ee" } };
    const layout = { ...baseLayout, yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" } };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotShortAppCounts() {
    const container = document.getElementById("short-app-counts");
    if (!container) return;
    const counts = data.short_app_counts;
    if (!counts || counts.length === 0) return;
    const labels = counts.map(row => row.app_name);
    const values = counts.map(row => row.count);
    const trace = {
      x: values,
      y: labels,
      type: "bar",
      orientation: "h",
      marker: { color: "#a78bfa" },
    };
    const layout = {
      ...baseLayout,
      margin: { l: 140, r: 10, t: 10, b: 40 },
      xaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" },
      yaxis: { automargin: true },
    };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  ensurePlotly(() => {
    plotLengthCounts();
    plotShortLabelCounts();
    plotShortWordCounts();
    plotShortRatingCounts();
    plotShortAppCounts();
  });
})();
