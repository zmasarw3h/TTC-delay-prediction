const PRESETS = {
  bus: {
    label: "Bus incident",
    payload: {
      mode: "bus",
      Route: "29",
      Direction: "N",
      Incident: "Mechanical",
      Location: "Dufferin Station",
      timestamp: "2024-02-03T08:30",
      prior_route_mean_delay: 10.0,
      prior_route_hour_mean_delay: 12.0,
      prior_incident_mean_delay: 9.0,
      prior_mode_mean_delay: 8.0,
      prior_global_mean_delay: 7.0,
      prior_route_hour_7d_mean_delay: 11.0,
    },
  },
  streetcar: {
    label: "Streetcar incident",
    payload: {
      mode: "streetcar",
      Route: "501",
      Direction: "E",
      Incident: "Operations",
      Location: "Queen St West at Spadina Ave",
      timestamp: "2024-09-18T17:45",
      prior_route_mean_delay: 13.5,
      prior_route_hour_mean_delay: 16.0,
      prior_incident_mean_delay: 11.5,
      prior_mode_mean_delay: 10.0,
      prior_global_mean_delay: 8.4,
      prior_route_hour_7d_mean_delay: 14.2,
    },
  },
};

const FIELD_NAMES = [
  "mode",
  "Route",
  "Direction",
  "Incident",
  "Location",
  "timestamp",
  "prior_route_mean_delay",
  "prior_route_hour_mean_delay",
  "prior_incident_mean_delay",
  "prior_mode_mean_delay",
  "prior_global_mean_delay",
  "prior_route_hour_7d_mean_delay",
];

const NUMERIC_FIELDS = new Set([
  "prior_route_mean_delay",
  "prior_route_hour_mean_delay",
  "prior_incident_mean_delay",
  "prior_mode_mean_delay",
  "prior_global_mean_delay",
  "prior_route_hour_7d_mean_delay",
]);

const statusEl = document.querySelector("#model-status");
const artifactPill = document.querySelector("#artifact-pill");
const form = document.querySelector("#prediction-form");
const resultsEl = document.querySelector("#results");
const submitButton = document.querySelector("#submit-button");
const presetButtons = document.querySelectorAll("[data-preset]");

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMinutes(value) {
  return `${Number(value).toFixed(1)} min`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function labelForFlag(value) {
  return Number(value) === 1 ? "Yes" : "No";
}

function setArtifactPill(health) {
  const exists = Boolean(health && health.artifact_exists);
  const loaded = Boolean(health && health.model_artifact_loaded);
  artifactPill.textContent = exists ? (loaded ? "Loaded" : "Available") : "Missing";
  artifactPill.className = `pill ${exists ? "ok" : "error"}`;
}

function statusItem(label, value) {
  return `
    <div class="status-item">
      <span class="label">${escapeHtml(label)}</span>
      <span class="value">${value}</span>
    </div>
  `;
}

function riskCutoffs(modelInfo) {
  const cutoffs = modelInfo.selected_operating_cutoffs || {};
  return (modelInfo.risk_thresholds || [])
    .map((threshold) => {
      const value = cutoffs[String(threshold)]?.probability_cutoff;
      return value === undefined ? `${threshold}+: unavailable` : `${threshold}+: ${formatPercent(value)}`;
    })
    .join("<br />");
}

function renderStatus(health, modelInfo) {
  const artifactStatus = health.artifact_exists
    ? health.model_artifact_loaded
      ? "Loaded in memory"
      : "Available on disk"
    : "Artifact not found";

  statusEl.classList.remove("results-empty");
  statusEl.innerHTML = [
    statusItem("Model", escapeHtml(modelInfo.model_name || "Unavailable")),
    statusItem("Phase", escapeHtml(modelInfo.model_phase || "Unavailable")),
    statusItem("Risk cutoffs", riskCutoffs(modelInfo) || "Unavailable"),
    statusItem("Artifact", escapeHtml(artifactStatus)),
  ].join("");
}

function renderStatusError(error) {
  artifactPill.textContent = "Unavailable";
  artifactPill.className = "pill error";
  statusEl.innerHTML = `
    <div class="error-box">
      <h3>Model status unavailable</h3>
      <div>${escapeHtml(error.message)}</div>
    </div>
  `;
}

async function loadModelStatus() {
  try {
    const [healthResponse, infoResponse] = await Promise.all([fetch("/health"), fetch("/model-info")]);
    const health = await healthResponse.json();
    setArtifactPill(health);

    if (!healthResponse.ok) {
      throw new Error(health.detail || "Health endpoint failed.");
    }
    if (!infoResponse.ok) {
      const detail = await infoResponse.json();
      throw new Error(detail.detail || "Model metadata endpoint failed.");
    }

    const modelInfo = await infoResponse.json();
    renderStatus(health, modelInfo);
  } catch (error) {
    renderStatusError(error);
  }
}

function setPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  for (const fieldName of FIELD_NAMES) {
    const input = document.getElementById(fieldName);
    if (input) {
      input.value = preset.payload[fieldName] ?? "";
    }
  }

  presetButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.preset === name);
  });
}

function buildPayload() {
  const payload = {};
  for (const fieldName of FIELD_NAMES) {
    const input = document.getElementById(fieldName);
    if (!input) continue;

    const value = input.value.trim();
    if (value === "") continue;
    payload[fieldName] = NUMERIC_FIELDS.has(fieldName) ? Number(value) : value;
  }
  return payload;
}

function resultCard(label, value, extraClass = "") {
  return `
    <div class="result-card ${extraClass}">
      <span class="label">${escapeHtml(label)}</span>
      <span class="value">${value}</span>
    </div>
  `;
}

function bandMarkup(band) {
  const normalized = String(band || "unknown").toLowerCase();
  return `<span class="band ${normalized}">${normalized}</span>`;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) return "";
  const items = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  return `
    <div class="warnings">
      <h3>Warnings</h3>
      <ul>${items}</ul>
    </div>
  `;
}

function renderResults(result) {
  resultsEl.className = "";
  resultsEl.innerHTML = `
    <div class="result-grid">
      ${resultCard("Predicted delay", formatMinutes(result.predicted_delay_minutes), "primary")}
      ${resultCard("Calibrated 30+ probability", formatPercent(result.calibrated_severe_delay_probability_30))}
      ${resultCard("Risk band 30", bandMarkup(result.risk_band_30))}
      ${resultCard("Severe delay prediction 30", labelForFlag(result.severe_delay_prediction_30))}
      ${resultCard("Calibrated 60+ probability", formatPercent(result.calibrated_severe_delay_probability_60))}
      ${resultCard("Risk band 60", bandMarkup(result.risk_band_60))}
      ${resultCard("Severe delay prediction 60", labelForFlag(result.severe_delay_prediction_60))}
    </div>
    ${renderWarnings(result.warnings)}
  `;
}

function renderError(message) {
  resultsEl.className = "";
  resultsEl.innerHTML = `
    <div class="error-box">
      <h3>Prediction failed</h3>
      <div>${escapeHtml(message)}</div>
    </div>
  `;
}

async function submitPrediction(event) {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";
  resultsEl.className = "results-empty";
  resultsEl.textContent = "Submitting engineered incident-time features...";

  try {
    const response = await fetch("/predict-delay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const body = await response.json();

    if (!response.ok) {
      throw new Error(body.detail || "Prediction endpoint returned an error.");
    }

    renderResults(body);
    loadModelStatus();
  } catch (error) {
    renderError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict delay";
  }
}

presetButtons.forEach((button) => {
  button.addEventListener("click", () => setPreset(button.dataset.preset));
});

form.addEventListener("submit", submitPrediction);
setPreset("bus");
loadModelStatus();
