(() => {
  const dataEl = document.getElementById("app-data");
  let appData = { category_pools: {}, umap_points: [] };

  if (dataEl && dataEl.textContent) {
    try {
      appData = JSON.parse(dataEl.textContent);
    } catch (err) {
      console.warn("Failed to parse app data payload.", err);
    }
  }

  const pools = appData.category_pools || {};
  const umapPoints = appData.umap_points || [];
  const groupSelect = document.getElementById("group-select");
  const categorySelect = document.getElementById("category-select");
  const sampleList = document.getElementById("sample-list");
  const clusterLabel = document.getElementById("cluster-label");
  const overlay = document.getElementById("loading-overlay");
  const progressBar = document.getElementById("progress-bar");
  let progressTimer;

  function startLoading() {
    if (!overlay || !progressBar) return;
    overlay.classList.remove("hidden");
    let pct = 5;
    progressBar.style.width = pct + "%";
    clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      pct = Math.min(95, pct + Math.random() * 8);
      progressBar.style.width = pct + "%";
    }, 400);
  }

  document.querySelectorAll("form").forEach(form => {
    form.addEventListener("submit", () => {
      startLoading();
    });
  });

  if (groupSelect && categorySelect && sampleList && clusterLabel) {
    function renderGroups() {
      const current = groupSelect.value;
      groupSelect.innerHTML = "";
      const groups = Object.keys(pools || {}).sort();
      groups.forEach(g => {
        const opt = document.createElement("option");
        opt.value = g;
        opt.textContent = g;
        groupSelect.appendChild(opt);
      });
      if (groups.length === 0) {
        const opt = document.createElement("option");
        opt.textContent = "Aucun groupe";
        opt.value = "";
        groupSelect.appendChild(opt);
        groupSelect.disabled = true;
      } else {
        groupSelect.disabled = false;
        groupSelect.value = groups.includes(current) ? current : groups[0];
      }
    }

    function renderCategories() {
      const current = categorySelect.value;
      categorySelect.innerHTML = "";
      const group = groupSelect.value;
      const list = pools[group] || [];
      list.forEach(entry => {
        const opt = document.createElement("option");
        opt.value = entry.category;
        opt.textContent = entry.category;
        categorySelect.appendChild(opt);
      });
      if (list.length === 0) {
        const opt = document.createElement("option");
        opt.textContent = "Aucune cat\u00e9gorie";
        opt.value = "";
        categorySelect.appendChild(opt);
        categorySelect.disabled = true;
      } else {
        categorySelect.disabled = false;
        const categories = list.map(entry => entry.category);
        categorySelect.value = categories.includes(current) ? current : categories[0];
      }
    }

    function drawSamples() {
      sampleList.innerHTML = "";
      const group = groupSelect.value;
      const category = categorySelect.value;
      const list = pools[group] || [];
      const entry = list.find(e => e.category === category);
      if (!entry || !entry.samples || entry.samples.length === 0) {
        sampleList.innerHTML = "<div class='sample'>Aucun extrait disponible.</div>";
        clusterLabel.textContent = "-";
        return;
      }
      clusterLabel.textContent = entry.cluster_ids ? entry.cluster_ids.join(", ") : "-";
      const picks = [];
      for (let i = 0; i < 10; i++) {
        const idx = Math.floor(Math.random() * entry.samples.length);
        picks.push(entry.samples[idx]);
      }
      picks.forEach(text => {
        const div = document.createElement("div");
        div.className = "sample";
        div.textContent = text;
        sampleList.appendChild(div);
      });
    }

    groupSelect.addEventListener("change", () => {
      renderCategories();
      drawSamples();
    });
    categorySelect.addEventListener("change", drawSamples);

    const refreshButton = document.getElementById("refresh-samples");
    if (refreshButton) {
      refreshButton.addEventListener("click", drawSamples);
    }

    renderGroups();
    renderCategories();
    drawSamples();
  }

  if (umapPoints && umapPoints.length > 0) {
    function ensurePlotly(cb) {
      if (window.Plotly) return cb();
      const s = document.createElement("script");
      s.src = "https://cdn.plot.ly/plotly-2.27.0.min.js";
      s.onload = cb;
      document.head.appendChild(s);
    }

    function hashColor(cluster) {
      const str = String(cluster);
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
      }
      const h = hash % 360;
      return `hsl(${h},70%,55%)`;
    }

    ensurePlotly(() => {
      const xs = umapPoints.map(p => p.x);
      const ys = umapPoints.map(p => p.y);
      const clusters = umapPoints.map(p => p.cluster);
      const colors = clusters.map(hashColor);
      const trace = {
        x: xs,
        y: ys,
        text: clusters.map(c => `Cluster ${c}`),
        mode: "markers",
        type: "scattergl",
        marker: { color: colors, size: 5, opacity: 0.7 },
        hovertemplate: "Cluster %{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
      };
      const layout = {
        margin: { l: 30, r: 10, t: 10, b: 30 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { visible: false },
        yaxis: { visible: false },
      };
      const config = {
        displaylogo: false,
        responsive: true,
        scrollZoom: true,
        modeBarButtonsToRemove: ["toggleSpikelines", "select2d", "lasso2d"],
      };
      layout.dragmode = "zoom";
      layout.hovermode = "closest";
      Plotly.newPlot("umap-plot", [trace], layout, config);
    });
  }
})();
