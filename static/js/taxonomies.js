(() => {
  const dataEl = document.getElementById("taxonomies-data");
  let data = { main_counts: [], taxonomy_breakdown: {}, rating_breakdown: null, weekly_trends: null };

  if (dataEl && dataEl.textContent) {
    try {
      data = JSON.parse(dataEl.textContent);
    } catch (err) {
      console.warn("Failed to parse taxonomies payload.", err);
    }
  }

  const overlay = document.getElementById("loading-overlay");
  const progressBar = document.getElementById("progress-bar");
  let progressTimer;

  function startLoading() {
    if (!overlay || !progressBar) return;
    overlay.classList.remove("hidden");
    let pct = 6;
    progressBar.style.width = pct + "%";
    clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      pct = Math.min(96, pct + Math.random() * 8);
      progressBar.style.width = pct + "%";
    }, 420);
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
    hoverlabel: { bgcolor: "#0b1220", bordercolor: "#38bdf8", font: { color: "#e2e8f0" } },
  };
  const baseConfig = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["toggleSpikelines", "select2d", "lasso2d"],
  };

  function plotMainCounts() {
    const container = document.getElementById("main-bar");
    if (!container) return;
    const mainCounts = data.main_counts || [];
    if (mainCounts.length === 0) return;

    const categories = mainCounts.map(row => row.category);
    const values = mainCounts.map(row => row.count);
    const trace = {
      x: values,
      y: categories,
      orientation: "h",
      type: "bar",
      marker: { color: "#38bdf8" },
    };
    const layout = {
      ...baseLayout,
      margin: { l: 140, r: 10, t: 10, b: 40 },
      xaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" },
      yaxis: { automargin: true },
    };
    Plotly.newPlot(container, [trace], layout, baseConfig);
  }

  function plotTaxonomyBreakdown() {
    const container = document.getElementById("taxonomy-breakdown");
    if (!container) return;
    const breakdown = data.taxonomy_breakdown || {};
    const mainCategories = breakdown.main_categories || [];
    const taxonomyCategories = breakdown.taxonomy_categories || [];
    const matrix = breakdown.matrix || [];
    if (mainCategories.length === 0 || taxonomyCategories.length === 0 || matrix.length === 0) return;

    const totals = taxonomyCategories.map((_, idx) =>
      matrix.reduce((sum, row) => sum + (row[idx] || 0), 0)
    );
    const ordered = totals
      .map((value, idx) => ({ idx, value }))
      .sort((a, b) => b.value - a.value);
    const topLimit = 12;
    const topIndices = ordered.slice(0, topLimit).map(row => row.idx);
    const hasRemainder = taxonomyCategories.length > topLimit;
    const reducedCategories = topIndices.map(idx => taxonomyCategories[idx]);
    if (hasRemainder) reducedCategories.push("Autres");

    const reducedMatrix = mainCategories.map((_, rowIdx) => {
      const row = matrix[rowIdx] || [];
      const values = topIndices.map(idx => row[idx] || 0);
      if (!hasRemainder) return values;
      const remainder = Math.max(0, 100 - values.reduce((sum, val) => sum + val, 0));
      return values.concat(remainder);
    });

    const traces = reducedCategories.map((category, idx) => ({
      name: category,
      x: mainCategories,
      y: reducedMatrix.map(row => row[idx] || 0),
      type: "bar",
    }));
    const layout = {
      barmode: "stack",
      ...baseLayout,
      margin: { l: 50, r: 10, t: 10, b: 140 },
      xaxis: { tickangle: -25, automargin: true, tickfont: { size: 11 } },
      yaxis: { title: "%", gridcolor: "rgba(148,163,184,0.2)" },
      legend: { orientation: "h", y: -0.35, font: { size: 10 }, title: { text: "Taxonomies" } },
    };
    Plotly.newPlot(container, traces, layout, baseConfig);
  }

  function plotRatingBreakdown() {
    const container = document.getElementById("rating-breakdown");
    if (!container) return;
    const breakdown = data.rating_breakdown;
    if (!breakdown) return;
    const mainCategories = breakdown.main_categories || [];
    const ratings = breakdown.ratings || [];
    const matrix = breakdown.matrix || [];
    if (mainCategories.length === 0 || ratings.length === 0 || matrix.length === 0) return;

    const traces = ratings.map((rating, idx) => ({
      name: rating,
      x: mainCategories,
      y: matrix.map(row => row[idx] || 0),
      type: "bar",
    }));
    const layout = {
      barmode: "stack",
      ...baseLayout,
      margin: { l: 50, r: 10, t: 10, b: 120 },
      xaxis: { tickangle: -25, automargin: true, tickfont: { size: 11 } },
      yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" },
      legend: { orientation: "h", y: -0.3, title: { text: "Rating" }, font: { size: 10 } },
    };
    Plotly.newPlot(container, traces, layout, baseConfig);
  }

  function plotWeeklyTrends() {
    const container = document.getElementById("weekly-trends");
    if (!container) return;
    const trends = data.weekly_trends;
    if (!trends) return;
    const weeks = trends.weeks || [];
    const categories = trends.categories || [];
    const matrix = trends.matrix || [];
    if (weeks.length === 0 || categories.length === 0 || matrix.length === 0) return;

    const traces = categories.map((category, idx) => ({
      name: category,
      x: weeks,
      y: matrix.map(row => row[idx] || 0),
      mode: "lines+markers",
      type: "scatter",
      line: { width: 2 },
    }));
    const layout = {
      ...baseLayout,
      margin: { l: 50, r: 10, t: 10, b: 110 },
      xaxis: { title: "Semaine", tickangle: -25, automargin: true, tickfont: { size: 11 } },
      yaxis: { title: "Avis", gridcolor: "rgba(148,163,184,0.2)" },
      legend: { orientation: "h", y: -0.3, font: { size: 10 } },
      hovermode: "x unified",
    };
    Plotly.newPlot(container, traces, layout, baseConfig);
  }

  if (data.main_counts && data.main_counts.length > 0) {
    ensurePlotly(() => {
      plotMainCounts();
      plotTaxonomyBreakdown();
      plotRatingBreakdown();
      plotWeeklyTrends();
    });
  }
})();
