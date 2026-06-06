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

const MODE_LABELS = {
  bus: "Bus",
  streetcar: "Streetcar",
};

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

const BASIC_FIELD_NAMES = [
  "mode",
  "Route",
  "Direction",
  "Incident",
  "Location",
  "timestamp",
];

const HISTORICAL_FIELD_NAMES = [
  "prior_route_mean_delay",
  "prior_route_hour_mean_delay",
  "prior_incident_mean_delay",
  "prior_mode_mean_delay",
  "prior_global_mean_delay",
  "prior_route_hour_7d_mean_delay",
  "prior_route_incident_mean_delay",
  "prior_mode_incident_mean_delay",
  "prior_route_direction_mean_delay",
  "prior_route_incident_count",
  "prior_route_30d_mean_delay",
  "prior_incident_30d_mean_delay",
  "prior_route_30d_severe_rate_30",
  "prior_incident_30d_severe_rate_30",
  "prior_route_30d_severe_rate_60",
  "prior_incident_30d_severe_rate_60",
  "prior_location_mean_delay",
  "prior_location_count",
];

const FIELD_NAMES = [...BASIC_FIELD_NAMES, ...HISTORICAL_FIELD_NAMES];
const NUMERIC_FIELDS = new Set(HISTORICAL_FIELD_NAMES);

const serviceNote = document.querySelector("#service-note");
const form = document.querySelector("#prediction-form");
const resultsEl = document.querySelector("#results");
const submitButton = document.querySelector("#submit-button");
const modeButtons = document.querySelectorAll("[data-mode]");
const routeInput = document.querySelector("#Route");
const routeMenu = document.querySelector("#route-menu");
const modeInput = document.querySelector("#mode");
const locationStatus = document.querySelector("#location-status");
const locationInput = document.querySelector("#Location");
const locationMenu = document.querySelector("#location-menu");
const historicalOverrideToggle = document.querySelector("#use-historical-overrides");

let routeOptions = [];
let routeStopDataAvailable = false;
let selectedRoute = null;
let routeLocations = [];
let selectedRouteLocation = null;
let routeLocationValidation = null;
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

function setDirectionOptions(options = DIRECTION_OPTIONS) {
  const values = setSelectOptions("Direction", options, DIRECTION_OPTIONS);
  directionValues = new Set(values);
  const select = document.getElementById("Direction");
  if (select && !directionValues.has(select.value)) {
    select.value = values[0] || "";
  }
}

function errorMessage(detail, fallback) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join(" ");
  }
  if (typeof detail === "string") return detail;
  return fallback;
}

function routeLabel(option) {
  return option.label && option.label !== option.value ? option.label : option.value;
}

function setMode(mode) {
  if (mode && modeValues.has(mode)) {
    modeInput.value = mode;
  } else {
    modeInput.value = "";
  }
  modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === modeInput.value);
    button.setAttribute("aria-pressed", String(button.dataset.mode === modeInput.value));
  });
}

function setHistoricalOverrideState() {
  const enabled = Boolean(historicalOverrideToggle?.checked);
  HISTORICAL_FIELD_NAMES.forEach((fieldName) => {
    const input = document.getElementById(fieldName);
    if (input) input.disabled = !enabled;
  });
}

function routeMatchesMode(option) {
  return !option.mode || !modeInput.value || option.mode === modeInput.value;
}

function setComboOpen(input, menu, isOpen) {
  input.setAttribute("aria-expanded", String(isOpen));
  menu.classList.toggle("open", isOpen);
}

function renderRouteMenu(showAll = false) {
  const query = showAll ? "" : routeInput.value.trim().toLowerCase();
  const filtered = routeOptions
    .filter(routeMatchesMode)
    .filter((option) => `${option.value} ${option.label || ""}`.toLowerCase().includes(query))
    .slice(0, 120);
  routeMenu.innerHTML = filtered.length
    ? filtered
        .map(
          (option) => `
            <button class="combo-option" type="button" data-route="${escapeHtml(option.value)}">
              ${escapeHtml(routeLabel(option))}
              ${option.mode ? `<span class="muted">${escapeHtml(MODE_LABELS[option.mode] || option.mode)}</span>` : ""}
            </button>
          `,
        )
        .join("")
    : '<div class="combo-empty">No matching routes</div>';
  setComboOpen(routeInput, routeMenu, true);
}

function renderLocationMenu() {
  const query = locationInput.value.trim().toLowerCase();
  const filtered = routeLocations
    .filter((option) => `${option.label} ${option.value}`.toLowerCase().includes(query))
    .slice(0, 160);
  locationMenu.innerHTML = filtered.length
    ? filtered
        .map(
          (option) => `
            <button class="combo-option" type="button" data-location="${escapeHtml(option.value)}">
              ${escapeHtml(option.label)}
              <span class="muted">${escapeHtml(option.value)}</span>
            </button>
          `,
        )
        .join("")
    : '<div class="combo-empty">No stops on this route match</div>';
  setComboOpen(locationInput, locationMenu, true);
}

async function loadServiceReadiness() {
  try {
    const [healthResponse, infoResponse, optionsResponse, routeOptionsResponse, historicalResponse] = await Promise.all([
      fetch("/health"),
      fetch("/model-info"),
      fetch("/model-options"),
      fetch("/route-options"),
      fetch("/historical-lookup-info"),
    ]);
    const health = await healthResponse.json();
    const info = await infoResponse.json();
    const options = await optionsResponse.json();
    const routePayload = await routeOptionsResponse.json();
    const historical = await historicalResponse.json();

    if (!healthResponse.ok) throw new Error(errorMessage(health.detail, "Health check failed."));
    if (!infoResponse.ok) throw new Error(errorMessage(info.detail, "Model metadata failed."));
    if (!optionsResponse.ok) throw new Error(errorMessage(options.detail, "Model options failed."));
    if (!routeOptionsResponse.ok) throw new Error(errorMessage(routePayload.detail, "Route options failed."));
    if (!historicalResponse.ok) throw new Error(errorMessage(historical.detail, "Historical lookup info failed."));

    modeValues = new Set((options.modes || MODE_OPTIONS).map((option) => normalizeOption(option).value));
    setDirectionOptions(options.directions || DIRECTION_OPTIONS);
    const incidentOptions =
      options.incidents && options.incidents.length
        ? options.incidents
        : CURATED_INCIDENTS.map((value) => ({ value, label: value }));
    incidentValues = new Set(setSelectOptions("Incident", incidentOptions));
    routeOptions = (routePayload.routes || []).map((option) => ({
      value: String(option.value),
      label: String(option.label || option.value),
      mode: option.mode || null,
    }));
    routeStopDataAvailable = Boolean(routePayload.gtfs_available);

    const systemsConnected = health.artifact_exists && routeStopDataAvailable && historical.loadable;
    serviceNote.textContent = systemsConnected
      ? "Systems connected."
      : "Some local systems are unavailable.";
    serviceNote.className = `service-note ${systemsConnected ? "ok" : "warn"}`;
  } catch (error) {
    setSelectOptions("Direction", DIRECTION_OPTIONS);
    setSelectOptions(
      "Incident",
      CURATED_INCIDENTS.map((value) => ({ value, label: value })),
    );
    serviceNote.textContent = `Service readiness unavailable: ${error.message}`;
    serviceNote.className = "service-note warn";
  }
}

function resetRouteLocationState(message) {
  routeLocations = [];
  selectedRouteLocation = null;
  routeLocationValidation = null;
  locationMatch = null;
  locationInput.value = "";
  locationInput.disabled = true;
  locationMenu.innerHTML = "";
  setComboOpen(locationInput, locationMenu, false);
  locationStatus.className = "location-status muted";
  locationStatus.textContent = message || "Choose a route first, then choose a stop on that route.";
}

function resetSelectedLocation(message) {
  selectedRouteLocation = null;
  routeLocationValidation = null;
  locationMatch = null;
  locationStatus.className = "location-status muted";
  locationStatus.textContent = message || "Choose a stop from the selected route.";
}

async function selectRoute(option) {
  selectedRoute = option;
  routeInput.value = option.value;
  setComboOpen(routeInput, routeMenu, false);
  clearValidationErrors();
  resetRouteLocationState("Loading stops for selected route...");
  await loadRouteLocations(option.value);
}

async function loadRouteLocations(routeValue) {
  if (!routeValue) {
    resetRouteLocationState();
    return;
  }

  try {
    const response = await fetch(`/route-locations?route=${encodeURIComponent(routeValue)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(errorMessage(body.detail, "Route locations failed."));
    routeStopDataAvailable = Boolean(body.gtfs_available);
    routeLocations = body.locations || [];
    if (body.directions && body.directions.length) {
      setDirectionOptions(body.directions);
    } else {
      setDirectionOptions(DIRECTION_OPTIONS);
    }
    if (!routeStopDataAvailable) {
      locationInput.disabled = true;
      locationStatus.className = "location-status warn";
      locationStatus.textContent = body.warning || "Route-stop validation is not available.";
      return;
    }
    if (!routeLocations.length) {
      locationInput.disabled = true;
      locationStatus.className = "location-status warn";
      locationStatus.textContent = body.warning || "No stops are available for the selected route.";
      return;
    }

    locationInput.disabled = false;
    locationStatus.className = body.warning ? "location-status suggest" : "location-status muted";
    locationStatus.textContent = body.warning || "Choose a stop from the selected route.";
  } catch (error) {
    routeLocations = [];
    locationInput.disabled = true;
    locationStatus.className = "location-status warn";
    locationStatus.textContent = `Could not load stops for this route: ${error.message}`;
  }
}

function selectLocation(option) {
  selectedRouteLocation = option;
  routeLocationValidation = null;
  locationMatch = null;
  locationInput.value = option.label;
  setComboOpen(locationInput, locationMenu, false);
  locationStatus.className = "location-status ok";
  locationStatus.innerHTML = `Selected route stop: <strong>${escapeHtml(option.label)}</strong>`;
  clearValidationErrors();
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

function exactRouteFromInput() {
  const value = routeInput.value.trim();
  return routeOptions.find((option) => routeMatchesMode(option) && (option.value === value || option.label === value)) || null;
}

function exactLocationFromInput() {
  const value = locationInput.value.trim().toLowerCase();
  return routeLocations.find((option) => option.label.toLowerCase() === value || option.value.toLowerCase() === value);
}

async function validateBasicForm() {
  clearValidationErrors();
  const errors = {};
  const requiredFields = {
    Route: "Choose a route.",
    Direction: "Choose a direction.",
    Incident: "Choose an incident type.",
    Location: "Choose a stop on the selected route.",
    timestamp: "Choose a timestamp.",
  };

  for (const [fieldName, message] of Object.entries(requiredFields)) {
    const value = document.getElementById(fieldName).value.trim();
    if (!value) errors[fieldName] = message;
  }

  if (!selectedRoute) {
    const exactRoute = exactRouteFromInput();
    if (exactRoute) {
      await selectRoute(exactRoute);
    } else if (routeInput.value.trim()) {
      errors.Route = "Choose a route from the route list.";
    }
  }

  const direction = document.getElementById("Direction").value;
  const incident = document.getElementById("Incident").value;
  const mode = modeInput.value;
  if (mode && !modeValues.has(mode)) errors.mode = "Mode could not be derived from the route.";
  if (!mode) errors.mode = "Mode could not be derived from the route.";
  if (direction && !directionValues.has(direction)) errors.Direction = "Choose a listed direction.";
  if (incident && !incidentValues.has(incident)) errors.Incident = "Choose a listed incident type.";

  if (!routeStopDataAvailable) {
    errors.Location = "Route-stop validation is unavailable; configure TTC GTFS data before prediction.";
  } else if (!selectedRouteLocation) {
    const exactLocation = exactLocationFromInput();
    if (exactLocation) {
      selectLocation(exactLocation);
    } else if (locationInput.value.trim()) {
      errors.Location = "Choose a stop from the selected route list.";
    }
  }

  for (const [fieldName, message] of Object.entries(errors)) {
    setFieldError(fieldName, message);
  }

  return Object.keys(errors).length === 0;
}

async function requestRouteLocationValidation() {
  if (!selectedRoute || !selectedRouteLocation) return null;
  try {
    const response = await fetch("/validate-route-location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        route: selectedRoute.value,
        location: selectedRouteLocation.value,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(errorMessage(body.detail, "Route-location validation failed."));
    routeLocationValidation = body;
    if (!body.accepted_for_prediction) {
      setFieldError("Location", body.warning || "Selected location is not on the selected route.");
      locationStatus.className = "location-status warn";
      locationStatus.textContent = body.warning || "Selected location is not on the selected route.";
      return null;
    }
    locationStatus.className = "location-status ok";
    locationStatus.innerHTML = `Validated route stop: <strong>${escapeHtml(body.route_location_label || body.route_location)}</strong>`;
    return body;
  } catch (error) {
    setFieldError("Location", error.message);
    locationStatus.className = "location-status warn";
    locationStatus.textContent = error.message;
    return null;
  }
}

function buildPayload() {
  const payload = {};
  const fieldsToSubmit = historicalOverrideToggle?.checked ? FIELD_NAMES : BASIC_FIELD_NAMES;
  for (const fieldName of fieldsToSubmit) {
    const input = document.getElementById(fieldName);
    if (!input) continue;

    const value = input.value.trim();
    if (value === "") continue;
    payload[fieldName] = NUMERIC_FIELDS.has(fieldName) ? Number(value) : value;
  }

  if (routeLocationValidation && routeLocationValidation.route_location) {
    payload.Location = routeLocationValidation.route_location;
  }
  if (locationMatch && locationMatch.accepted_for_prediction && locationMatch.matched_location) {
    payload.Location = locationMatch.matched_location;
  }
  return payload;
}

async function requestLocationMatch() {
  const location = routeLocationValidation?.route_location || selectedRouteLocation?.value || locationInput.value.trim();
  if (!location) return null;

  try {
    const response = await fetch("/match-location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ location }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(errorMessage(body.detail, "Location matching failed."));
    locationMatch = body;
    return body;
  } catch (error) {
    locationMatch = null;
    locationStatus.className = "location-status suggest";
    locationStatus.textContent = `Route stop validated. Model location matching unavailable: ${error.message}`;
    return null;
  }
}

async function ensureRouteLocationForPrediction() {
  if (routeLocationValidation?.accepted_for_prediction) return routeLocationValidation;
  return requestRouteLocationValidation();
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
  if (!(await validateBasicForm())) {
    resultsEl.className = "results-empty";
    resultsEl.textContent = "Fix the highlighted fields before estimating delay.";
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Estimating...";
  resultsEl.className = "results-empty";
  resultsEl.textContent = "Validating route and location...";

  try {
    const routeValidation = await ensureRouteLocationForPrediction();
    if (!routeValidation?.accepted_for_prediction) {
      resultsEl.className = "results-empty";
      resultsEl.textContent = "Choose a location from the selected route before estimating delay.";
      return;
    }

    await requestLocationMatch();
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

routeInput.addEventListener("focus", () => {
  routeInput.select();
  renderRouteMenu(true);
});
routeInput.addEventListener("input", () => {
  selectedRoute = null;
  resetRouteLocationState("Choose a route from the route list.");
  setDirectionOptions(DIRECTION_OPTIONS);
  renderRouteMenu();
});
routeInput.addEventListener("blur", () => {
  window.setTimeout(() => setComboOpen(routeInput, routeMenu, false), 120);
});
routeMenu.addEventListener("click", (event) => {
  const button = event.target.closest("[data-route]");
  if (!button) return;
  const option = routeOptions.find((candidate) => candidate.value === button.dataset.route);
  if (option) void selectRoute(option);
});

locationInput.addEventListener("focus", () => {
  if (!locationInput.disabled) renderLocationMenu();
});
locationInput.addEventListener("input", () => {
  resetSelectedLocation("Choose a stop from the selected route list.");
  renderLocationMenu();
});
locationInput.addEventListener("blur", () => {
  window.setTimeout(() => setComboOpen(locationInput, locationMenu, false), 120);
});
locationMenu.addEventListener("click", (event) => {
  const button = event.target.closest("[data-location]");
  if (!button) return;
  const option = routeLocations.find((candidate) => candidate.value === button.dataset.location);
  if (option) selectLocation(option);
});

modeButtons.forEach((button) => {
  button.setAttribute("aria-pressed", String(button.classList.contains("active")));
  button.addEventListener("click", () => {
    if (button.dataset.mode === modeInput.value) return;
    setMode(button.dataset.mode);
    selectedRoute = null;
    routeInput.value = "";
    resetRouteLocationState("Choose a route from the route list.");
    setDirectionOptions(DIRECTION_OPTIONS);
    renderRouteMenu(true);
  });
});

form.addEventListener("submit", submitPrediction);
historicalOverrideToggle?.addEventListener("change", setHistoricalOverrideState);

setDirectionOptions(DIRECTION_OPTIONS);
setSelectOptions(
  "Incident",
  CURATED_INCIDENTS.map((value) => ({ value, label: value })),
);
setMode("bus");
setHistoricalOverrideState();
resetRouteLocationState();
void loadServiceReadiness();
