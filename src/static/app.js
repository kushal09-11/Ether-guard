/**
 * ChainSentinel — Phase 4 Demonstration Controller
 * Communicates with the FastAPI backend prediction service, handles client-side address
 * format validation, renders risk score gauges, cold-start warnings, zero-activity states,
 * and handles Etherscan API errors with readable guidance.
 */

// Feature metadata dictionary with human-readable units, categories, and parity classifications
const FEATURE_META = {
  "avg_min_between_sent_tnx": { category: "Temporal", unit: "minutes", parity: "Approximate" },
  "avg_min_between_received_tnx": { category: "Temporal", unit: "minutes", parity: "Approximate" },
  "time_diff_first_last_mins": { category: "Temporal", unit: "minutes", parity: "Approximate" },
  "sent_tnx": { category: "Volume", unit: "transactions", parity: "Approximate" },
  "received_tnx": { category: "Volume", unit: "transactions", parity: "Approximate" },
  "created_contracts_count": { category: "Volume", unit: "contracts", parity: "Approximate" },
  "unique_received_from_addresses": { category: "Diversity", unit: "addresses", parity: "Approximate" },
  "unique_sent_to_addresses": { category: "Diversity", unit: "addresses", parity: "Approximate" },
  "min_value_received_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "max_value_received_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "avg_value_received_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "min_value_sent_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "max_value_sent_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "avg_value_sent_eth": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "total_transactions_count": { category: "Volume", unit: "transactions", parity: "Approximate" },
  "total_ether_sent": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "total_ether_received": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "total_ether_balance": { category: "Ether Flow", unit: "ETH", parity: "Unverifiable" },
  "erc20_total_tnx": { category: "ERC-20", unit: "transfers", parity: "Unverifiable" },
  "erc20_unique_sent_addr": { category: "ERC-20", unit: "addresses", parity: "Unverifiable" },
  "erc20_unique_received_addr": { category: "ERC-20", unit: "addresses", parity: "Unverifiable" },
  "erc20_unique_received_contract_addr": { category: "ERC-20", unit: "contracts", parity: "Unverifiable" },
  "erc20_unique_sent_token_name": { category: "ERC-20", unit: "symbols", parity: "Unverifiable" },
  "erc20_unique_received_token_name": { category: "ERC-20", unit: "symbols", parity: "Unverifiable" },
};

const ETH_ADDRESS_REGEX = /^0x[a-fA-F0-9]{40}$/;
const GAUGE_CIRCUMFERENCE = 314.159; // 2 * PI * 50

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const addressInput = document.getElementById("addressInput");
  const clearInputBtn = document.getElementById("clearInputBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeSpinner = document.getElementById("analyzeSpinner");
  const analyzeForm = document.getElementById("analyzeForm");
  const validationMsg = document.getElementById("validationMsg");
  const presetChips = document.getElementById("presetChips");

  const alertBanner = document.getElementById("alertBanner");
  const alertTitle = document.getElementById("alertTitle");
  const alertMessage = document.getElementById("alertMessage");
  const alertHelp = document.getElementById("alertHelp");
  const retryBtn = document.getElementById("retryBtn");
  const dismissAlertBtn = document.getElementById("dismissAlertBtn");

  const loadingState = document.getElementById("loadingState");
  const resultsSection = document.getElementById("resultsSection");
  const coldStartCard = document.getElementById("coldStartCard");
  const coldStartText = document.getElementById("coldStartText");
  const zeroActivityCard = document.getElementById("zeroActivityCard");
  const scoreableContent = document.getElementById("scoreableContent");

  const gaugeCircle = document.getElementById("gaugeCircle");
  const riskScoreNumber = document.getElementById("riskScoreNumber");
  const decisionBadge = document.getElementById("decisionBadge");
  const riskLevelBadge = document.getElementById("riskLevelBadge");
  const decisionNote = document.getElementById("decisionNote");

  const statNormalTxs = document.getElementById("statNormalTxs");
  const statErc20Txs = document.getElementById("statErc20Txs");
  const statLifespan = document.getElementById("statLifespan");
  const statBalance = document.getElementById("statBalance");
  const statEthSent = document.getElementById("statEthSent");
  const statEthRecv = document.getElementById("statEthRecv");
  const displayAddress = document.getElementById("displayAddress");
  const historyCompleteBadge = document.getElementById("historyCompleteBadge");

  const toggleFeaturesBtn = document.getElementById("toggleFeaturesBtn");
  const featuresWrapper = document.getElementById("featuresWrapper");
  const featuresTableBody = document.getElementById("featuresTableBody");
  const disclaimersList = document.getElementById("disclaimersList");

  const openModelInfoBtn = document.getElementById("openModelInfoBtn");
  const closeModelInfoBtn = document.getElementById("closeModelInfoBtn");
  const modelModalBackdrop = document.getElementById("modelModalBackdrop");
  const modalBody = document.getElementById("modalBody");
  const apiKeyDot = document.getElementById("apiKeyDot");
  const apiKeyText = document.getElementById("apiKeyText");
  const apiKeyStatusPill = document.getElementById("apiKeyStatusPill");

  // Last executed action for Retry support and concurrency guard
  let lastAction = null;
  let isAnalyzing = false;
  let consecutiveRateLimits = 0;
  let retryCooldownTimer = null;

  // Initialize System Health & API Key Status
  fetchHealth();

  // -------------------------------------------------------------
  // Address Input & Validation
  // -------------------------------------------------------------
  function validateAddress(value) {
    const trimmed = (value || "").trim();
    if (!trimmed) {
      validationMsg.textContent = "";
      validationMsg.className = "input-validation-msg";
      return false;
    }
    if (ETH_ADDRESS_REGEX.test(trimmed)) {
      validationMsg.textContent = "✓ Valid Ethereum wallet address format.";
      validationMsg.className = "input-validation-msg valid";
      return true;
    } else {
      if (!trimmed.startsWith("0x")) {
        validationMsg.textContent = "Address must begin with '0x'.";
      } else if (trimmed.length !== 42) {
        validationMsg.textContent = `Address length is ${trimmed.length} characters (expected 42 characters).`;
      } else {
        validationMsg.textContent = "Address contains invalid non-hexadecimal characters.";
      }
      validationMsg.className = "input-validation-msg invalid";
      return false;
    }
  }

  addressInput.addEventListener("input", () => {
    const val = addressInput.value.trim();
    clearInputBtn.style.display = val.length > 0 ? "block" : "none";
    validateAddress(val);
    clearActivePreset();
  });

  clearInputBtn.addEventListener("click", () => {
    addressInput.value = "";
    clearInputBtn.style.display = "none";
    validationMsg.textContent = "";
    validationMsg.className = "input-validation-msg";
    clearActivePreset();
    addressInput.focus();
  });

  function clearActivePreset() {
    document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
  }

  // -------------------------------------------------------------
  // Form Submission (Analyze Wallet)
  // -------------------------------------------------------------
  analyzeForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const address = addressInput.value.trim();
    if (!validateAddress(address)) {
      addressInput.focus();
      return;
    }
    runWalletAnalysis(address);
  });

  // -------------------------------------------------------------
  // Preset Chips Handling
  // -------------------------------------------------------------
  presetChips.addEventListener("click", (e) => {
    const btn = e.target.closest(".preset-chip");
    if (!btn) return;
    const presetId = btn.getAttribute("data-preset");
    if (!presetId) return;

    clearActivePreset();
    btn.classList.add("active");

    if (presetId === "live_vitalik") {
      addressInput.value = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045";
      clearInputBtn.style.display = "block";
      validateAddress(addressInput.value);
      runWalletAnalysis(addressInput.value);
    } else {
      runPresetAnalysis(presetId);
    }
  });

  // -------------------------------------------------------------
  // API Calls: Health Check
  // -------------------------------------------------------------
  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      if (data.etherscan_configured) {
        apiKeyDot.className = "status-dot success";
        apiKeyText.textContent = "Etherscan API: Configured";
        apiKeyStatusPill.title = "ETHERSCAN_API_KEY detected in environment. Live on-chain analysis active.";
      } else {
        apiKeyDot.className = "status-dot warning";
        apiKeyText.textContent = "Etherscan API: Offline Mode";
        apiKeyStatusPill.title = "No ETHERSCAN_API_KEY detected. Use demonstration presets or set key in .env for live analysis.";
      }
    } catch (err) {
      apiKeyDot.className = "status-dot danger";
      apiKeyText.textContent = "Backend Offline";
      apiKeyStatusPill.title = "Unable to connect to local FastAPI backend.";
    }
  }

  // -------------------------------------------------------------
  // API Calls: Execute Wallet Analysis
  // -------------------------------------------------------------
  async function runWalletAnalysis(address) {
    if (isAnalyzing) {
      console.warn("Analysis already running. Ignoring duplicate request.");
      return;
    }
    isAnalyzing = true;
    lastAction = () => runWalletAnalysis(address);
    showLoading(true);
    hideAlert();
    resultsSection.style.display = "none";

    try {
      const response = await fetch("/api/assess-wallet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ address, max_pages: 10 }),
      });

      const data = await response.json();

      if (!response.ok) {
        handleApiError(response.status, data);
        return;
      }

      consecutiveRateLimits = 0;
      renderAssessment(data);
    } catch (err) {
      showAlert(
        "Network Failure",
        "Unable to communicate with the local ChainSentinel server.",
        "Ensure the backend process is running locally on http://127.0.0.1:8000."
      );
    } finally {
      isAnalyzing = false;
      showLoading(false);
    }
  }

  // -------------------------------------------------------------
  // API Calls: Execute Preset Analysis
  // -------------------------------------------------------------
  async function runPresetAnalysis(presetId) {
    if (isAnalyzing) {
      console.warn("Analysis already running. Ignoring duplicate preset trigger.");
      return;
    }
    isAnalyzing = true;
    lastAction = () => runPresetAnalysis(presetId);
    showLoading(true);
    hideAlert();
    resultsSection.style.display = "none";

    try {
      const response = await fetch(`/api/assess-preset/${presetId}`, {
        method: "POST",
      });

      const data = await response.json();

      if (!response.ok) {
        handleApiError(response.status, data);
        return;
      }

      // Sync input field to the preset's address
      addressInput.value = data.address || "";
      clearInputBtn.style.display = addressInput.value ? "block" : "none";
      validateAddress(addressInput.value);

      consecutiveRateLimits = 0;
      renderAssessment(data);
    } catch (err) {
      showAlert(
        "Network Failure",
        "Failed to load demonstration preset from backend.",
        "Ensure the backend process is running."
      );
    } finally {
      isAnalyzing = false;
      showLoading(false);
    }
  }

  // -------------------------------------------------------------
  // Cooldown & Error Handling
  // -------------------------------------------------------------
  function startRetryCooldown(seconds = 3) {
    if (retryCooldownTimer) {
      clearInterval(retryCooldownTimer);
      retryCooldownTimer = null;
    }
    let remaining = seconds;
    retryBtn.disabled = true;
    retryBtn.textContent = `Retry in ${remaining}s...`;

    retryCooldownTimer = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(retryCooldownTimer);
        retryCooldownTimer = null;
        retryBtn.disabled = false;
        retryBtn.textContent = "Retry Analysis";
      } else {
        retryBtn.textContent = `Retry in ${remaining}s...`;
      }
    }, 1000);
  }

  function handleApiError(status, data) {
    let title = "Assessment Error";
    let message = "An error occurred while evaluating the wallet.";
    let help = "";

    const detail = data.detail;
    if (typeof detail === "object" && detail !== null) {
      title = detail.error_type || title;
      message = detail.message || message;
      help = detail.help || "";
    } else if (typeof detail === "string") {
      message = detail;
    }

    if (status === 503) {
      if (typeof detail === "object" && detail?.error_type === "EtherscanQuotaExceededError") {
        title = "Etherscan Daily Request Quota Exhausted";
        help = help || "Your Etherscan API key daily quota (100k requests/day) has been reached. Please wait for the daily reset or use offline presets.";
      } else {
        title = "Etherscan API Key Required for Live Analysis";
        help = help || "Add your ETHERSCAN_API_KEY to .env or use the offline demonstration presets above.";
      }
    } else if (status === 429) {
      consecutiveRateLimits += 1;
      if (consecutiveRateLimits > 1) {
        title = "Etherscan Rate Limit Persisting";
        message = "Upstream Etherscan rate limit is still active after multiple attempts.";
        help = "The shared free-tier limit (5 req/s) is currently saturated. Please wait 10–15 seconds before retrying.";
        startRetryCooldown(5);
      } else {
        title = "Etherscan Rate Limit Exceeded";
        help = help || "Free-tier Etherscan keys allow up to 5 requests per second. The client retried automatically with backoff. Please wait a few seconds.";
        startRetryCooldown(3);
      }
    } else if (status === 504) {
      title = "Etherscan Gateway Timeout";
      help = help || "Etherscan API took too long to reply. Click Retry to re-attempt.";
    }

    showAlert(title, message, help);
  }

  function showAlert(title, message, help) {
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    alertHelp.textContent = help || "";
    alertBanner.style.display = "flex";
  }

  function hideAlert() {
    alertBanner.style.display = "none";
  }

  dismissAlertBtn.addEventListener("click", hideAlert);

  retryBtn.addEventListener("click", () => {
    if (isAnalyzing || retryBtn.disabled) return;
    if (typeof lastAction === "function") {
      lastAction();
    }
  });

  // -------------------------------------------------------------
  // UI State Controls
  // -------------------------------------------------------------
  function showLoading(isLoading) {
    analyzeBtn.disabled = isLoading;
    addressInput.disabled = isLoading;
    document.querySelectorAll(".preset-chip").forEach(c => c.disabled = isLoading);
    if (isLoading) {
      analyzeSpinner.style.display = "inline-block";
      loadingState.style.display = "flex";
      retryBtn.disabled = true;
    } else {
      analyzeSpinner.style.display = "none";
      loadingState.style.display = "none";
      if (!retryCooldownTimer) {
        retryBtn.disabled = false;
      }
    }
  }

  // -------------------------------------------------------------
  // Render Assessment Results
  // -------------------------------------------------------------
  function renderAssessment(data) {
    resultsSection.style.display = "block";
    displayAddress.textContent = data.address || "Unknown Address";

    // Update Analysis Mode Badge
    const modeBadge = document.getElementById("analysisModeBadge");
    const zeroModeNote = document.getElementById("zeroActivityModeNote");

    if (data.analysis_mode === "LIVE_ON_CHAIN") {
      if (modeBadge) {
        modeBadge.textContent = "Live On-Chain (Etherscan V2)";
        modeBadge.className = "mode-badge live";
      }
      if (zeroModeNote) zeroModeNote.textContent = "Source: Live on-chain retrieval via Etherscan API V2 (0 records found)";
    } else if (data.analysis_mode === "OFFLINE_PRESET") {
      const pName = data.preset_name || "Offline Scenario";
      if (modeBadge) {
        modeBadge.textContent = `Demo Preset: ${pName}`;
        modeBadge.className = "mode-badge preset";
      }
      if (zeroModeNote) zeroModeNote.textContent = `Source: Offline demonstration scenario (${pName})`;
    } else {
      if (modeBadge) {
        modeBadge.textContent = "Custom Features Input";
        modeBadge.className = "mode-badge custom";
      }
      if (zeroModeNote) zeroModeNote.textContent = "Source: Custom feature vector";
    }

    // 1. Zero Activity / Insufficient Activity Wallet
    if (!data.is_scoreable && data.decision === "INSUFFICIENT_ACTIVITY") {
      zeroActivityCard.style.display = "block";
      scoreableContent.style.display = "none";
      coldStartCard.style.display = "none";
      return;
    }

    // 2. Scoreable Wallet
    zeroActivityCard.style.display = "none";
    scoreableContent.style.display = "block";

    // Cold-Start Warning Card
    if (data.has_cold_start) {
      coldStartCard.style.display = "flex";
      // Find cold start warning text
      const coldWarning = (data.warnings || []).find(w => w.includes("COLD-START WARNING")) ||
        "Wallet lifespan is under 24 hours or has <= 1 transaction. Tree models heavily penalize short lifespans, elevating false-positive risk.";
      coldStartText.textContent = coldWarning;
    } else {
      coldStartCard.style.display = "none";
    }

    // Risk Score Dial & Badges
    const score = typeof data.risk_score === "number" ? data.risk_score : 0.0;
    riskScoreNumber.textContent = score.toFixed(4);

    // Gauge circle animation
    const offset = GAUGE_CIRCUMFERENCE * (1 - Math.min(Math.max(score, 0), 1));
    gaugeCircle.style.strokeDashoffset = offset;

    // Dial color styling
    gaugeCircle.className = "gauge-fill " + (score >= 0.75 ? "high" : score >= (data.decision_threshold || 0.51) ? "med" : "low");

    // Decision Badge
    if (data.decision === "FLAGGED") {
      decisionBadge.textContent = "FLAGGED";
      decisionBadge.className = "decision-badge flagged";
      decisionNote.textContent = `Score exceeds operational validation threshold (${data.decision_threshold || 0.51}).`;
    } else {
      decisionBadge.textContent = "NORMAL";
      decisionBadge.className = "decision-badge normal";
      decisionNote.textContent = `Score is below operational validation threshold (${data.decision_threshold || 0.51}).`;
    }

    // Risk Level Badge
    const rLevel = data.risk_level || "LOW";
    riskLevelBadge.textContent = `${rLevel} RISK`;
    riskLevelBadge.className = "risk-level-badge " + (rLevel === "HIGH" ? "high" : rLevel === "MEDIUM" ? "med" : "low");

    // Activity Indicators
    const feats = data.features || {};
    statNormalTxs.textContent = (data.raw_normal_count !== undefined && data.raw_normal_count !== null) ? data.raw_normal_count : (feats.total_transactions_count || 0);
    statErc20Txs.textContent = (data.raw_erc20_count !== undefined && data.raw_erc20_count !== null) ? data.raw_erc20_count : (feats.erc20_total_tnx || 0);

    const lifespanMins = feats.time_diff_first_last_mins || 0;
    if (lifespanMins < 60) {
      statLifespan.textContent = `${lifespanMins.toFixed(1)} mins`;
    } else if (lifespanMins < 1440) {
      statLifespan.textContent = `${(lifespanMins / 60).toFixed(1)} hrs`;
    } else {
      statLifespan.textContent = `${(lifespanMins / 1440).toFixed(1)} days`;
    }

    statBalance.textContent = formatEth(feats.total_ether_balance || 0);
    statEthSent.textContent = formatEth(feats.total_ether_sent || 0);
    statEthRecv.textContent = formatEth(feats.total_ether_received || 0);

    // History completeness badge
    if (data.is_history_complete === false) {
      historyCompleteBadge.textContent = "Partial (Pagination Truncated)";
      historyCompleteBadge.className = "badge warning";
    } else {
      historyCompleteBadge.textContent = "Complete History";
      historyCompleteBadge.className = "badge success";
    }

    // Populate 24 Feature Breakdown Table
    populateFeaturesTable(feats);

    // Populate Disclaimers
    populateDisclaimers(data.disclaimers || []);
  }

  function formatEth(val) {
    if (Math.abs(val) < 0.0001 && val !== 0) {
      return val.toExponential(2) + " ETH";
    }
    return Number(val).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) + " ETH";
  }

  function populateFeaturesTable(feats) {
    featuresTableBody.innerHTML = "";
    const featureKeys = Object.keys(FEATURE_META);

    featureKeys.forEach((key) => {
      const meta = FEATURE_META[key] || { category: "General", unit: "", parity: "Approximate" };
      const rawVal = feats[key] !== undefined ? feats[key] : 0;
      const formattedVal = typeof rawVal === "number" ? (Number.isInteger(rawVal) ? rawVal : rawVal.toFixed(4)) : rawVal;

      const row = document.createElement("tr");

      const tdName = document.createElement("td");
      tdName.innerHTML = `<code class="feature-code">${key}</code>`;

      const tdCat = document.createElement("td");
      tdCat.textContent = meta.category;

      const tdVal = document.createElement("td");
      tdVal.className = "feature-val";
      tdVal.textContent = formattedVal;

      const tdUnit = document.createElement("td");
      tdUnit.textContent = meta.unit;

      const tdParity = document.createElement("td");
      const parityClass = meta.parity === "Approximate" ? "approximate" : meta.parity === "Verified" ? "verified" : "unverifiable";
      tdParity.innerHTML = `<span class="parity-badge ${parityClass}">${meta.parity}</span>`;

      row.appendChild(tdName);
      row.appendChild(tdCat);
      row.appendChild(tdVal);
      row.appendChild(tdUnit);
      row.appendChild(tdParity);

      featuresTableBody.appendChild(row);
    });
  }

  function populateDisclaimers(disclaimers) {
    disclaimersList.innerHTML = "";
    disclaimers.forEach((d) => {
      const li = document.createElement("li");
      li.textContent = d;
      disclaimersList.appendChild(li);
    });
  }

  // Toggle Features Visibility
  toggleFeaturesBtn.addEventListener("click", () => {
    if (featuresWrapper.style.display === "none") {
      featuresWrapper.style.display = "block";
      toggleFeaturesBtn.textContent = "Collapse Features";
    } else {
      featuresWrapper.style.display = "none";
      toggleFeaturesBtn.textContent = "Expand Features";
    }
  });

  // -------------------------------------------------------------
  // Model Specification & Audit Modal
  // -------------------------------------------------------------
  openModelInfoBtn.addEventListener("click", async () => {
    modelModalBackdrop.style.display = "flex";
    modalBody.innerHTML = `<div class="loading-spinner" style="margin: 20px auto;"></div>`;

    try {
      const res = await fetch("/api/model-info");
      const info = await res.json();
      renderModelInfoModal(info);
    } catch (err) {
      modalBody.innerHTML = `<p style="color: var(--color-danger);">Failed to load model specifications from API.</p>`;
    }
  });

  closeModelInfoBtn.addEventListener("click", () => {
    modelModalBackdrop.style.display = "none";
  });

  modelModalBackdrop.addEventListener("click", (e) => {
    if (e.target === modelModalBackdrop) {
      modelModalBackdrop.style.display = "none";
    }
  });

  function renderModelInfoModal(info) {
    const cm = info.held_out_test_metrics.confusion_matrix;
    const m = info.held_out_test_metrics;

    modalBody.innerHTML = `
      <section>
        <h4 style="color: #fff; margin-bottom: 8px;">Model Architecture</h4>
        <div class="metric-row"><span class="metric-row-name">Pipeline Model:</span><span class="metric-row-val">${info.model_type}</span></div>
        <div class="metric-row"><span class="metric-row-name">Hyperparameters:</span><span class="metric-row-val">max_depth=6, min_samples_leaf=20, random_state=42</span></div>
        <div class="metric-row"><span class="metric-row-name">Input Schema:</span><span class="metric-row-val">24 Canonical Features</span></div>
        <div class="metric-row"><span class="metric-row-name">Operational Threshold:</span><span class="metric-row-val">T = ${info.held_out_test_metrics.decision_threshold} (Tuned on Val Split)</span></div>
      </section>

      <section>
        <h4 style="color: #fff; margin-bottom: 8px;">Held-Out Test Set Performance (N=1,473)</h4>
        <div class="metric-row"><span class="metric-row-name">Confusion Matrix:</span><span class="metric-row-val">TN=${cm.true_negatives}, FP=${cm.false_positives}, FN=${cm.false_negatives}, TP=${cm.true_positives}</span></div>
        <div class="metric-row"><span class="metric-row-name">Precision (Fraud):</span><span class="metric-row-val">${(m.precision * 100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-row-name">Recall (Fraud):</span><span class="metric-row-val">${(m.recall * 100).toFixed(2)}%</span></div>
        <div class="metric-row"><span class="metric-row-name">F1 Score (Fraud):</span><span class="metric-row-val">${m.fraud_f1.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-row-name">Macro F1 Score:</span><span class="metric-row-val">${m.macro_f1.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-row-name">PR-AUC:</span><span class="metric-row-val">${m.pr_auc.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-row-name">ROC-AUC:</span><span class="metric-row-val">${m.roc_auc.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-row-name">Brier Score:</span><span class="metric-row-val">${m.brier_score.toFixed(4)} (Probabilistic error metric, not ranking)</span></div>
      </section>

      <section>
        <h4 style="color: #fff; margin-bottom: 8px;">Feature Compatibility Parity Status</h4>
        <div class="metric-row"><span class="metric-row-name">Verified Equivalent:</span><span class="metric-row-val" style="color: var(--color-success);">${info.feature_parity_status.verified_equivalent} / 24</span></div>
        <div class="metric-row"><span class="metric-row-name">Approximate Equivalent:</span><span class="metric-row-val" style="color: var(--color-info);">${info.feature_parity_status.approximate_equivalent} / 24</span></div>
        <div class="metric-row"><span class="metric-row-name">Not Verifiable from Dataset:</span><span class="metric-row-val" style="color: var(--color-warning);">${info.feature_parity_status.not_verifiable_from_dataset} / 24</span></div>
        <p style="font-size: 0.8rem; color: var(--text-subtle); margin-top: 6px;">${info.feature_parity_status.note}</p>
      </section>
    `;
  }
});
