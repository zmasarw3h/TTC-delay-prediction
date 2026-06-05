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

const serviceNote = document.querySelector("#service-note");
const form = document.querySelector("#prediction-form");
const resultsEl = document.querySelector("#results");
const submitButton = document.querySelector("#submit-button");
const presetButtons = document.querySelectorAll("[data-preset]");
const matchLocationButton = document.querySelector("#match-location-button");
const locationStatus = document.querySelector("#location-status");
const locationInput = document.querySelector("#Location");

let locationMatch = null;

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

function setOptions(selectId, values, fallbackValues = []) {
  const select = document.getElementById(selectId);
  if (!select) return;
  const options = values && values.length ? values : fallbackValues;
  select.innerHTML = options
    .map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`)
    .join("");
}

function setDatalist(listId, values) {
  const list = document.getElementById(listId);
  if (!list || !values) return;
  list.innerHTML = values
    .slice(0, 1000)
    .map((value) => `<option value="${escapeHtml(value)}"></option>`)
    .join("");
}

async function loadServiceReadiness() {
  try {
    const [healthResponse, infoResponse, optionsResponse] = await Promise.all([
      fetch("/health"),
      fetch("/model-info"),
      fetch("/model-options"),
    ]);
    const health = await healthResponse.json();
    const info = await infoResponse.json();
    const options = await optionsResponse.json();

    if (!healthResponse.ok) throw new Error(health.detail || "Health check failed.");
    if (!infoResponse.ok) throw new Error(info.detail || "Model metadata failed.");
    if (!optionsResponse.ok) throw new Error(options.detail || "Model options failed.");

    setOptions("mode", options.modes, ["bus", "streetcar"]);
    setOptions("Direction", options.directions, ["N", "S", "E", "W", "B", "Unknown"]);
    setDatalist("route-options", options.routes);
    setDatalist("incident-options", options.incidents);
    setDatalist("location-options", options.locations);

    const warningText = options.warnings && options.warnings.length ? ` ${options.warnings.join(" ")}` : "";
    const artifactText = health.artifact_exists ? "Artifact available." : "Artifact missing.";
    serviceNote.textContent = `${artifactText} ${info.model_phase || "Model"} options loaded for route, incident, direction, and location guidance.${warningText}`;
    serviceNote.className = `service-note ${health.artifact_exists ? "ok" : "warn"}`;
    setPreset(activePresetName());
  } catch (error) {
    serviceNote.textContent = `Service readiness unavailable: ${error.message}`;
    serviceNote.className = "service-note warn";
  }
}

function activePresetName() {
  const active = Array.from(presetButtons).find((button) => button.classList.contains("active"));
  return active ? active.dataset.preset : "bus";
}

function resetLocationMatch(message) {
  locationMatch = null;
  locationStatus.className = "location-status muted";
  locationStatus.innerHTML = escapeHtml(message || "Location has not been matched yet.");
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
  resetLocationMatch("Preset loaded. Match the location or submit to use the entered location.");
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

  if (locationMatch && locationMatch.accepted_for_prediction && locationMatch.matched_location) {
    payload.Location = locationMatch.matched_location;
  }
  return payload;
}

async function requestLocationMatch() {
  const location = locationInput.value.trim();
  if (!location) {
    resetLocationMatch("Enter a location before matching.");
    return null;
  }

  locationStatus.className = "location-status muted";
  locationStatus.textContent = "Matching location...";

  try {
    const response = await fetch("/match-location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "Location matching failed.");
    renderLocationMatch(body);
    return body;
  } catch (error) {
    locationMatch = null;
    locationStatus.className = "location-status warn";
    locationStatus.textContent = error.message;
    return null;
  }
}

function renderLocationMatch(match) {
  locationMatch = match;
  if (match.accepted_for_prediction && match.matched_location) {
    locationStatus.className = "location-status ok";
    locationStatus.innerHTML = `Matched to: <strong>${escapeHtml(match.matched_location)}</strong>`;
    return;
  }

  if (match.matched_location && match.match_type === "fuzzy") {
    locationStatus.className = "location-status suggest";
    locationStatus.innerHTML = `
      Suggested: <strong>${escapeHtml(match.matched_location)}</strong>
      <button id="accept-location-button" class="inline-button" type="button">Accept suggestion</button>
    `;
    document.querySelector("#accept-location-button").addEventListener("click", () => {
      locationInput.value = match.matched_location;
      locationMatch = { ...match, accepted_for_prediction: true };
      locationStatus.className = "location-status ok";
      locationStatus.innerHTML = `Matched to: <strong>${escapeHtml(match.matched_location)}</strong>`;
    });
    return;
  }

  locationStatus.className = "location-status warn";
  locationStatus.textContent = "No confident match; using entered location.";
}

async function ensureLocationMatch() {
  if (
    locationMatch &&
    locationMatch.original_location === locationInput.value.trim() &&
    (locationMatch.accepted_for_prediction || locationMatch.match_type === "none")
  ) {
    return locationMatch;
  }
  return requestLocationMatch();
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
  return `<span class="band ${normalized}">${escapeHtml(normalized)}</span>`;
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) return "";
  const items = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  return `
    <div class="warnings">
      <h3>Input notes</h3>
      <ul>${items}</ul>
    </div>
  `;
}

function renderModelDetails(result) {
  return `
    <details class="model-details">
      <summary>Model details</summary>
      <dl>
        <div>
          <dt>30+ minute operating cutoff</dt>
          <dd>${formatPercent(result.selected_probability_cutoff_30)}</dd>
        </div>
        <div>
          <dt>60+ minute operating cutoff</dt>
          <dd>${formatPercent(result.selected_probability_cutoff_60)}</dd>
        </div>
      </dl>
    </details>
  `;
}

function renderResults(result) {
  resultsEl.className = "";
  resultsEl.innerHTML = `
    <div class="result-grid">
      ${resultCard("Expected delay", formatMinutes(result.predicted_delay_minutes), "primary")}
      ${resultCard("Chance of 30+ min delay", formatPercent(result.calibrated_severe_delay_probability_30))}
      ${resultCard("30+ min risk level", bandMarkup(result.risk_band_30))}
      ${resultCard("Flagged for 30+ min delay?", labelForFlag(result.severe_delay_prediction_30))}
      ${resultCard("Chance of 60+ min delay", formatPercent(result.calibrated_severe_delay_probability_60))}
      ${resultCard("60+ min risk level", bandMarkup(result.risk_band_60))}
      ${resultCard("Flagged for 60+ min delay?", labelForFlag(result.severe_delay_prediction_60))}
    </div>
    ${renderWarnings(result.warnings)}
    ${renderModelDetails(result)}
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
  submitButton.textContent = "Estimating...";
  resultsEl.className = "results-empty";
  resultsEl.textContent = "Submitting incident-time features...";

  try {
    await ensureLocationMatch();
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
  } catch (error) {
    renderError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Estimate delay";
  }
}

presetButtons.forEach((button) => {
  button.addEventListener("click", () => setPreset(button.dataset.preset));
});

matchLocationButton.addEventListener("click", requestLocationMatch);
locationInput.addEventListener("input", () => resetLocationMatch("Location changed. Match again or submit to use it as entered."));
locationInput.addEventListener("blur", () => {
  if (locationInput.value.trim()) requestLocationMatch();
});
form.addEventListener("submit", submitPrediction);
setPreset("bus");
loadServiceReadiness();
