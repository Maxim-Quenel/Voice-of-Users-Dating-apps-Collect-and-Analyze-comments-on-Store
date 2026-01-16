(() => {
  const thresholdInput = document.getElementById("rag_threshold");
  const thresholdValue = document.getElementById("threshold-value");
  const overlay = document.getElementById("loading-overlay");
  const progressBar = document.getElementById("progress-bar");
  let progressTimer;

  if (thresholdInput && thresholdValue) {
    thresholdInput.addEventListener("input", () => {
      thresholdValue.textContent = thresholdInput.value;
    });
  }

  function startLoading() {
    if (!overlay || !progressBar) return;
    overlay.classList.remove("hidden");
    let pct = 8;
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

  const chartDataEl = document.getElementById("rag-chart-data");
  let chartData = { columns: [], sample: [], types: {}, sample_size: 0 };

  if (chartDataEl && chartDataEl.textContent) {
    try {
      chartData = JSON.parse(chartDataEl.textContent);
    } catch (err) {
      console.warn("Failed to parse chart payload.", err);
    }
  }

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
    hoverlabel: { bgcolor: "#0b1220", bordercolor: "#22d3ee", font: { color: "#e2e8f0" } },
    margin: { l: 50, r: 10, t: 10, b: 60 },
  };
  const baseConfig = {
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ["toggleSpikelines", "select2d", "lasso2d"],
  };

  function buildOptions(select, options, includeEmpty) {
    if (!select) return;
    select.innerHTML = "";
    if (includeEmpty) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Aucun";
      select.appendChild(opt);
    }
    options.forEach(value => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value;
      select.appendChild(opt);
    });
  }

  function getColumnType(col) {
    return (chartData.types && chartData.types[col]) || "text";
  }

  function normalizeValue(value, type) {
    if (value === null || value === undefined) return null;
    if (type === "numeric") {
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }
    if (type === "date") {
      const parsed = new Date(value);
      if (Number.isNaN(parsed.getTime())) return null;
      return parsed.toISOString();
    }
    const text = String(value).trim();
    return text ? text : null;
  }

  function aggregateRows(rows, xCol, yCol, agg, colorCol) {
    const xType = getColumnType(xCol);
    const yType = getColumnType(yCol);
    const cType = colorCol ? getColumnType(colorCol) : "text";
    const buckets = new Map();

    rows.forEach(row => {
      const xVal = normalizeValue(row[xCol], xType);
      if (xVal === null) return;
      const colorVal = colorCol ? normalizeValue(row[colorCol], cType) : "Série";
      if (colorCol && colorVal === null) return;

      let yVal = 1;
      if (agg !== "count") {
        yVal = normalizeValue(row[yCol], yType);
        if (yVal === null) return;
      }
      const key = `${colorVal}||${xVal}`;
      if (!buckets.has(key)) {
        buckets.set(key, { color: colorVal, x: xVal, values: [], sum: 0, count: 0 });
      }
      const bucket = buckets.get(key);
      bucket.count += 1;
      if (agg !== "count") {
        bucket.sum += yVal;
        bucket.values.push(yVal);
      }
    });

    const totals = new Map();
    const seriesMap = new Map();
    buckets.forEach(bucket => {
      let y = bucket.count;
      if (agg === "sum") y = bucket.sum;
      if (agg === "avg") y = bucket.count ? bucket.sum / bucket.count : 0;
      if (agg === "median") {
        const sorted = bucket.values.slice().sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        y = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      }

      if (!seriesMap.has(bucket.color)) {
        seriesMap.set(bucket.color, new Map());
      }
      seriesMap.get(bucket.color).set(bucket.x, y);
      totals.set(bucket.x, (totals.get(bucket.x) || 0) + y);
    });

    return { seriesMap, totals, xType };
  }

  function sortKeys(keys, xType, totals) {
    if (xType === "numeric") {
      return keys.sort((a, b) => Number(a) - Number(b));
    }
    if (xType === "date") {
      return keys.sort((a, b) => new Date(a) - new Date(b));
    }
    return keys.sort((a, b) => (totals.get(b) || 0) - (totals.get(a) || 0));
  }

  function buildBarOrLine({ rows, xCol, yCol, agg, colorCol, limit, type }) {
    const { seriesMap, totals, xType } = aggregateRows(rows, xCol, yCol, agg, colorCol);
    let keys = Array.from(totals.keys());
    const isCategorical = xType === "text";
    if (isCategorical && limit) {
      keys = sortKeys(keys, xType, totals).slice(0, limit);
    } else {
      keys = sortKeys(keys, xType, totals);
    }

    const traces = [];
    seriesMap.forEach((map, color) => {
      const x = [];
      const y = [];
      keys.forEach(key => {
        if (!map.has(key)) return;
        x.push(key);
        y.push(map.get(key));
      });
      traces.push({
        x,
        y,
        type: type === "line" ? "scatter" : "bar",
        mode: type === "line" ? "lines+markers" : undefined,
        name: color,
      });
    });
    return traces;
  }

  function buildScatter(rows, xCol, yCol, colorCol) {
    const xType = getColumnType(xCol);
    const yType = getColumnType(yCol);
    const cType = colorCol ? getColumnType(colorCol) : "text";
    const groups = new Map();

    rows.forEach(row => {
      const xVal = normalizeValue(row[xCol], xType);
      const yVal = normalizeValue(row[yCol], yType);
      if (xVal === null || yVal === null) return;
      const colorVal = colorCol ? normalizeValue(row[colorCol], cType) : "Série";
      if (colorCol && colorVal === null) return;
      if (!groups.has(colorVal)) {
        groups.set(colorVal, { x: [], y: [] });
      }
      const group = groups.get(colorVal);
      group.x.push(xVal);
      group.y.push(yVal);
    });

    const traces = [];
    groups.forEach((group, name) => {
      traces.push({
        x: group.x,
        y: group.y,
        type: "scatter",
        mode: "markers",
        name,
      });
    });
    return traces;
  }

  function buildHistogram(rows, xCol, colorCol) {
    const xType = getColumnType(xCol);
    const cType = colorCol ? getColumnType(colorCol) : "text";
    const groups = new Map();

    rows.forEach(row => {
      const xVal = normalizeValue(row[xCol], xType);
      if (xVal === null) return;
      const colorVal = colorCol ? normalizeValue(row[colorCol], cType) : "Série";
      if (colorCol && colorVal === null) return;
      if (!groups.has(colorVal)) {
        groups.set(colorVal, []);
      }
      groups.get(colorVal).push(xVal);
    });

    const traces = [];
    groups.forEach((values, name) => {
      traces.push({ x: values, type: "histogram", name, opacity: 0.75 });
    });
    return traces;
  }

  function buildBox(rows, xCol, yCol, colorCol) {
    const xType = getColumnType(xCol);
    const yType = getColumnType(yCol);
    const cType = colorCol ? getColumnType(colorCol) : "text";
    const groups = new Map();

    rows.forEach(row => {
      const yVal = normalizeValue(row[yCol], yType);
      if (yVal === null) return;
      const xVal = xCol ? normalizeValue(row[xCol], xType) : null;
      if (xCol && xVal === null) return;
      const colorVal = colorCol ? normalizeValue(row[colorCol], cType) : "Série";
      if (colorCol && colorVal === null) return;
      if (!groups.has(colorVal)) {
        groups.set(colorVal, { x: [], y: [] });
      }
      const group = groups.get(colorVal);
      group.x.push(xVal);
      group.y.push(yVal);
    });

    const traces = [];
    groups.forEach((group, name) => {
      traces.push({
        x: group.x,
        y: group.y,
        type: "box",
        name,
      });
    });
    return traces;
  }

  function initChartBuilder() {
    if (!chartData.sample || chartData.sample.length === 0) return;

    const typeEl = document.getElementById("chart-type");
    const xEl = document.getElementById("chart-x");
    const yEl = document.getElementById("chart-y");
    const colorEl = document.getElementById("chart-color");
    const aggEl = document.getElementById("chart-agg");
    const limitEl = document.getElementById("chart-limit");
    const refreshEl = document.getElementById("chart-refresh");
    const hintEl = document.getElementById("chart-hint");
    const canvas = document.getElementById("chart-canvas");

    if (!typeEl || !xEl || !yEl || !colorEl || !aggEl || !limitEl || !refreshEl || !canvas) {
      return;
    }

    const columns = chartData.columns || [];
    const numericCols = columns.filter(col => getColumnType(col) === "numeric");
    const dateCols = columns.filter(col => getColumnType(col) === "date");

    buildOptions(xEl, columns, false);
    buildOptions(yEl, numericCols.length ? numericCols : columns, true);
    buildOptions(colorEl, columns, true);

    if (dateCols.length > 0) {
      xEl.value = dateCols[0];
    } else {
      xEl.value = columns[0] || "";
    }
    if (numericCols.length > 0) {
      yEl.value = numericCols[0];
    }

    function updateControls() {
      const type = typeEl.value;
      const isHistogram = type === "histogram";
      const isScatter = type === "scatter";
      const isBox = type === "box";
      const requiresY = !(isHistogram);
      yEl.disabled = !requiresY;
      aggEl.disabled = isScatter || isHistogram || isBox;
      if (isHistogram && yEl.value === "") {
        yEl.value = numericCols[0] || "";
      }
    }

    function renderChart() {
      if (!window.Plotly) return;
      const type = typeEl.value;
      const xCol = xEl.value;
      const yCol = yEl.value;
      const colorCol = colorEl.value || "";
      const agg = aggEl.value;
      const limit = parseInt(limitEl.value, 10) || 0;
      const rows = chartData.sample || [];

      let traces = [];
      let layout = { ...baseLayout };
      let error = "";

      if (type === "scatter") {
        if (!xCol || !yCol) {
          error = "Choisissez X et Y pour le nuage de points.";
        } else if (getColumnType(yCol) !== "numeric") {
          error = "Le nuage de points nécessite un Y numérique.";
        } else {
          traces = buildScatter(rows, xCol, yCol, colorCol);
        }
      } else if (type === "bar" || type === "line") {
        if (!xCol) {
          error = "Choisissez un axe X.";
        } else if (agg !== "count" && (!yCol || getColumnType(yCol) !== "numeric")) {
          error = "Choisissez un Y numérique pour cette agrégation.";
        } else {
          traces = buildBarOrLine({ rows, xCol, yCol, agg, colorCol, limit, type });
        }
      } else if (type === "histogram") {
        if (!xCol) {
          error = "Choisissez un axe X.";
        } else if (getColumnType(xCol) !== "numeric") {
          error = "L'histogramme nécessite un X numérique.";
        } else {
          traces = buildHistogram(rows, xCol, colorCol);
          layout.barmode = "overlay";
        }
      } else if (type === "box") {
        if (!yCol || getColumnType(yCol) !== "numeric") {
          error = "Le boxplot nécessite un Y numérique.";
        } else {
          traces = buildBox(rows, xCol, yCol, colorCol);
        }
      }

      if (hintEl) {
        hintEl.textContent = error || "Graphique prêt.";
      }

      if (traces.length === 0) {
        Plotly.purge(canvas);
        return;
      }

      Plotly.newPlot(canvas, traces, layout, baseConfig);
    }

    updateControls();
    typeEl.addEventListener("change", () => {
      updateControls();
      renderChart();
    });
    [xEl, yEl, colorEl, aggEl, limitEl].forEach(el => {
      el.addEventListener("change", renderChart);
    });
    refreshEl.addEventListener("click", renderChart);

    ensurePlotly(renderChart);
  }

  initChartBuilder();
})();
