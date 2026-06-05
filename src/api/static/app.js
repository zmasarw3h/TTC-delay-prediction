const CURATED_INCIDENTS = [
  "Mechanical",
  "Utilized Off Route",
  "General Delay",
  "Late Leaving Garage",
  "Investigation",
  "Operations - Operator",
  "Operations",
  "Diversion",
  "Emergency Services",
  "Security",
  "Collision - TTC",
  "Collision - TTC Involved",
  "Road Blocked - NON-TTC Collision",
  "Held By",
  "Cleaning",
  "Cleaning - Unsanitary",
  "Vision",
  "Overhead",
  "Overhead - Pantograph",
  "Rail/Switches",
  "Other",
  "Unknown",
];

const MODE_OPTIONS = [
  { value: "bus", label: "Bus" },
  { value: "streetcar", label: "Streetcar" },
];

const DIRECTION_OPTIONS = [
  { value: "N", label: "North" },
  { value: "E", label: "East" },
  { value: "S", label: "South" },
  { value: "W", label: "West" },
  { value: "B", label: "Both / bidirectional" },
  { value: "Unknown", label: "Unknown" },
];

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
let incidentValues = new Set(CURATED_INCIDENTS);
let modeValues = new Set(MODE_OPTIONS.map((option) => option.value));
let directionValues = new Set(DIRECTION_OPTIONS.map((option) => option.value));

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMinutes(value) {
  return `${Number(value).toFixed(1)} minutes`;
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

function normalizeOption(option) {
  if (typeof option === "string") return { value: option, label: option };
  return { value: option.value, label: option.label || option.value };
}

function setSelectOptions(selectId, options, fallbackOptions = []) {
  const select = document.getElementById(selectId);
  if (!select) return [];
  const normalized = (options && options.length ? options : fallbackOptions).map(normalizeOption);
  select.innerHTML = normalized
    .map((option) => `<option value="${escapeHtml(option.value)}">${escapeHtml(option.label)}</option>`)
    .join("");
  return normalized.map((option) => option.value);
}

function isRouteLike(value) {
  return /^(\d{1,4}[A-Za-z]{0,2}|RAD)$/.test(String(value).trim());
}

function setRouteOptions(values) {
  const list = document.getElementById("route-options");
  if (!list || !values) return;
  const routeValues = Array.from(new Set(values.map(String).filter(isRouteLike)));
  list.innerHTML = routeValues
    .slice(0, 1000)
    .map((value) => `<option value="${escapeHtml(value)}"></option>`)
    .join("");
}

function errorMessage(detail, fallback) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join(" ");
  }
  if (typeof detail === "string") return detail;
  return fallback;
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

    if (!healthResponse.ok) throw new Error(errorMessage(health.detail, "Health check failed."));
    if (!infoResponse.ok) throw new Error(errorMessage(info.detail, "Model metadata failed."));
    if (!optionsResponse.ok) throw new Error(errorMessage(options.detail, "Model options failed."));

    modeValues = new Set(setSelectOptions("mode", options.modes, MODE_OPTIONS));
    directionValues = new Set(setSelectOptions("Direction", options.directions, DIRECTION_OPTIONS));
    const incidentOptions =
      options.incidents && options.incidents.length
        ? options.incidents
        : CURATED_INCIDENTS.map((value) => ({ value, label: value }));
    incidentValues = new Set(setSelectOptions("Incident", incidentOptions));
    setRouteOptions(options.routes || []);

    const artifactText = health.artifact_exists ? "Model artifact available." : "Model artifact missing.";
    serviceNote.textContent = `${artifactText} Planner-ready category options loaded. Historical feature lookup is not implemented yet.`;
    serviceNote.className = `service-note ${health.artifact_exists ? "ok" : "warn"}`;
    setPreset(activePresetName());
  } catch (error) {
    setSelectOptions("mode", MODE_OPTIONS);
    setSelectOptions("Direction", DIRECTION_OPTIONS);
    setSelectOptions(
      "Incident",
      CURATED_INCIDENTS.map((value) => ({ value, label: value })),
    );
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
  clearValidationErrors();
  resetLocationMatch("Preset loaded. Match the location or submit to use the entered location.");
}

function clearValidationErrors() {
  document.querySelectorAll(".field-error").forEach((element) => {
    element.textContent = "";
  });
}

function setFieldError(fieldName, message) {
  const error = document.querySelector(`[data-error-for="${fieldName}"]`);
  if (error) error.textContent = message;
}

function validateForm() {
  clearValidationErrors();
  const errors = {};
  const requiredFields = {
    mode: "Choose a mode.",
    Route: "Enter a route.",
    Direction: "Choose a direction.",
    Incident: "Choose an incident type.",
    Location: "Enter a location.",
    timestamp: "Choose a timestamp.",
  };

  for (const [fieldName, message] of Object.entries(requiredFields)) {
    const value = document.getElementById(fieldName).value.trim();
    if (!value) errors[fieldName] = message;
  }

  const mode = document.getElementById("mode").value;
  const direction = document.getElementById("Direction").value;
  const incident = document.getElementById("Incident").value;
  if (mode && !modeValues.has(mode)) errors.mode = "Mode must be Bus or Streetcar.";
  if (direction && !directionValues.has(direction)) errors.Direction = "Choose a listed direction.";
  if (incident && !incidentValues.has(incident)) errors.Incident = "Choose a listed incident type.";

  for (const [fieldName, message] of Object.entries(errors)) {
    setFieldError(fieldName, message);
  }

  return Object.keys(errors).length === 0;
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
    if (!response.ok) throw new Error(errorMessage(body.detail, "Location matching failed."));
    renderLocationMatch(body);
    return body;
  } catch (error) {
    locationMatch = null;
    locationStatus.className = "location-status warn";
    locationStatus.textContent = `Location matching unavailable: ${error.message}. Using entered location.`;
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
      <button id="accept-location-button" class="inline-button" type="button">Use suggestion</button>
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
  locationStatus.textContent = match.warning || "No confident match; using entered location.";
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
  const label = normalized.charAt(0).toUpperCase() + normalized.slice(1);
  return `<span class="band ${normalized}">${escapeHtml(label)}</span>`;
}

function renderWarnings(warnings) {
  const notes = warnings && warnings.length ? warnings : ["No input warnings returned."];
  const items = notes.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("");
  return resultCard("Input notes", `<ul class="note-list">${items}</ul>`);
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
      ${resultCard("Chance of 30+ minute delay", formatPercent(result.calibrated_severe_delay_probability_30))}
      ${resultCard("30+ minute risk level", bandMarkup(result.risk_band_30))}
      ${resultCard("30+ minute delay flag", labelForFlag(result.severe_delay_prediction_30))}
      ${resultCard("Chance of 60+ minute delay", formatPercent(result.calibrated_severe_delay_probability_60))}
      ${resultCard("60+ minute risk level", bandMarkup(result.risk_band_60))}
      ${resultCard("60+ minute delay flag", labelForFlag(result.severe_delay_prediction_60))}
      ${renderWarnings(result.warnings)}
    </div>
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
  if (!validateForm()) {
    resultsEl.className = "results-empty";
    resultsEl.textContent = "Fix the highlighted fields before estimating delay.";
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Estimating...";
  resultsEl.className = "results-empty";
  resultsEl.textContent = "Submitting incident details...";

  try {
    await ensureLocationMatch();
    const response = await fetch("/predict-delay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const body = await response.json();

    if (!response.ok) {
      throw new Error(errorMessage(body.detail, "Prediction endpoint returned an error."));
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
form.addEventListener("submit", submitPrediction);

setSelectOptions("mode", MODE_OPTIONS);
setSelectOptions("Direction", DIRECTION_OPTIONS);
setSelectOptions(
  "Incident",
  CURATED_INCIDENTS.map((value) => ({ value, label: value })),
);
setPreset("bus");
loadServiceReadiness();
