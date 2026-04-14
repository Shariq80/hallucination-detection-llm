// ===== STATE =====
let currentResults = null;
let experimentData = {};
let generatedClaimsList = []; // Array of { claim, status, label, verifications }

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadExperimentResults();

  document.getElementById('topicInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') generateClaims();
  });
});

// ===== TABS =====
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ===== FILL TOPIC =====
function fillTopic(el) {
  document.getElementById('topicInput').value = el.textContent;
  document.getElementById('topicInput').focus();
}

// ===== GENERATE CLAIMS =====
async function generateClaims() {
  const input = document.getElementById('topicInput');
  const btn = document.getElementById('generateBtn');
  const topic = input.value.trim();

  if (!topic) {
    input.focus();
    input.style.borderColor = 'var(--accent-red)';
    setTimeout(() => input.style.borderColor = '', 1500);
    return;
  }

  // Show loading
  btn.classList.add('loading');
  btn.disabled = true;
  document.getElementById('claimsTableSection').classList.add('hidden');
  document.getElementById('claimDetailsSection').classList.add('hidden');

  try {
    const response = await fetch('/api/generate_claims', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic,
        num_claims: parseInt(document.getElementById('numClaimsSelect').value)
      })
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    generatedClaimsList = data.claims.map(c => ({
      claim: c,
      status: 'pending', // pending, running, complete, error
      label: '—',
      verifications: null
    }));

    renderMasterClaimsTable();

  } catch (err) {
    console.error('Generation failed:', err);
    alert('Failed to generate claims. Check server logs.');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ===== MASTER CLAIMS TABLE =====
function renderMasterClaimsTable() {
  const tbody = document.getElementById('masterClaimsBody');
  tbody.innerHTML = '';

  generatedClaimsList.forEach((item, idx) => {
    const tr = document.createElement('tr');
    tr.className = 'clickable';
    tr.onclick = () => selectClaim(idx);

    let statusHtml = '';
    if (item.status === 'pending') statusHtml = '<span style="color:var(--text-muted)">⏳ Pending</span>';
    else if (item.status === 'running') statusHtml = '<span style="color:var(--accent-cyan)">⏱️ Running...</span>';
    else if (item.status === 'error') statusHtml = '<span style="color:var(--accent-red)">⚠️ Error</span>';
    else statusHtml = '<span style="color:var(--accent-green)">✅ Complete</span>';

    // Badge formatting for final label
    let labelHtml = item.label;
    if (item.label !== '—') {
       labelHtml = `<span class="model-card__badge ${getBadgeClass(item.label)}">${item.label}</span>`;
    }

    tr.innerHTML = `
      <td style="color:var(--text-muted)">${idx + 1}</td>
      <td style="font-weight:500;">${item.claim.substring(0, 60)}${item.claim.length > 60 ? '...' : ''}</td>
      <td>${statusHtml}</td>
      <td>${labelHtml}</td>
      <td><span style="color:var(--accent-blue);font-size:0.8rem">View Details ➔</span></td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('claimsTableSection').classList.remove('hidden');
}

// ===== VERIFY ALL CLAIMS =====
async function verifyAllClaims() {
  const btn = document.getElementById('verifyAllBtn');
  btn.classList.add('loading');
  btn.disabled = true;
  document.getElementById('claimDetailsSection').classList.add('hidden');

  const topK = parseInt(document.getElementById('topKSelect').value);

  for (let i = 0; i < generatedClaimsList.length; i++) {
    if (generatedClaimsList[i].status === 'complete') continue;

    generatedClaimsList[i].status = 'running';
    renderMasterClaimsTable();

    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ claim: generatedClaimsList[i].claim, top_k: topK })
      });

      if (!response.ok) throw new Error('API Error');
      
      const data = await response.json();
      generatedClaimsList[i].status = 'complete';
      generatedClaimsList[i].verifications = data;
      
      // Calculate consensus label (if all same, else Mixed)
      const labels = data.model_results.map(m => m.label);
      const allSame = labels.every(l => l === labels[0]);
      generatedClaimsList[i].label = allSame ? labels[0] : 'MIXED';

    } catch (err) {
      console.error(err);
      generatedClaimsList[i].status = 'error';
    }

    renderMasterClaimsTable();
  }

  btn.classList.remove('loading');
  btn.disabled = false;
}

// ===== SELECT CLAIM =====
function selectClaim(idx) {
  const item = generatedClaimsList[idx];
  
  // Update table row highlighting
  const rows = document.getElementById('masterClaimsBody').querySelectorAll('tr');
  rows.forEach((r, i) => {
    if (i === idx) r.classList.add('active-row');
    else r.classList.remove('active-row');
  });

  if (item.status !== 'complete' || !item.verifications) {
    document.getElementById('claimDetailsSection').classList.add('hidden');
    return;
  }

  document.getElementById('selectedClaimText').innerText = item.claim;
  renderResults(item.verifications);
  document.getElementById('claimDetailsSection').classList.remove('hidden');
}

// ===== PROGRESS TRACKER =====
function showProgress() {
  const tracker = document.getElementById('progressTracker');
  tracker.classList.remove('hidden');
  const steps = tracker.querySelectorAll('.progress-step');
  steps.forEach(s => {
    s.classList.remove('active', 'done');
    s.querySelector('.status-dot').className = 'status-dot';
  });

  // Animate steps sequentially
  let delay = 0;
  steps.forEach((step, i) => {
    setTimeout(() => {
      step.classList.add('active');
      step.querySelector('.status-dot').classList.add('status-dot--loading');
    }, delay);
    delay += 800;
  });
}

function hideProgress() {
  const tracker = document.getElementById('progressTracker');
  const steps = tracker.querySelectorAll('.progress-step');
  steps.forEach(s => {
    s.classList.remove('active');
    s.classList.add('done');
    s.querySelector('.status-dot').className = 'status-dot status-dot--done';
  });
  setTimeout(() => tracker.classList.add('hidden'), 1500);
}

// ===== SKELETONS =====
function showSkeletons() {
  document.getElementById('skeletonGrid').classList.remove('hidden');
}

function hideSkeletons() {
  document.getElementById('skeletonGrid').classList.add('hidden');
}

function hideResults() {
  document.getElementById('resultsSection').classList.add('hidden');
  document.getElementById('evidenceSection').classList.add('hidden');
  document.getElementById('consensusBanner').classList.add('hidden');
  document.getElementById('terminalSection').classList.add('hidden');
}

// ===== RENDER RESULTS =====
function renderResults(data) {
  renderConsensus(data);
  renderModelCards(data);
  renderEvidenceTable(data);
  renderAggregationTable(data);
  renderTerminalLog(data);
}

// ===== CONSENSUS BANNER =====
function renderConsensus(data) {
  const banner = document.getElementById('consensusBanner');
  const labels = data.model_results.map(m => m.label);
  const allSame = labels.every(l => l === labels[0]);

  banner.className = allSame
    ? 'consensus-banner consensus-banner--agree'
    : 'consensus-banner consensus-banner--disagree';

  banner.innerHTML = allSame
    ? `✅ All 3 models agree: <strong>${labels[0]}</strong>`
    : `⚠️ Models disagree — review individual results below`;

  banner.classList.remove('hidden');
}

// ===== MODEL CARDS =====
function renderModelCards(data) {
  const grid = document.getElementById('modelGrid');
  grid.innerHTML = '';

  const modelColors = [
    { accent: 'var(--accent-blue)', gradient: 'linear-gradient(90deg, var(--accent-blue), var(--accent-purple))' },
    { accent: 'var(--accent-cyan)', gradient: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))' },
    { accent: 'var(--accent-purple)', gradient: 'linear-gradient(90deg, var(--accent-purple), var(--accent-orange))' },
  ];

  data.model_results.forEach((model, idx) => {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.style.setProperty('--card-accent', modelColors[idx]?.accent || 'var(--accent-blue)');

    const badgeClass = getBadgeClass(model.label);
    const nli = model.nli_scores || {};
    const sim = model.similarity_score || 0;

    card.innerHTML = `
      <div class="model-card__header">
        <div>
          <div class="model-card__name">${formatModelName(model.model_name)}</div>
        </div>
        <span class="model-card__badge ${badgeClass}">${model.label}</span>
      </div>
      <div class="score-bars">
        <div class="score-bar score-bar--entailment">
          <div class="score-bar__header">
            <span class="score-bar__label">Entailment</span>
            <span class="score-bar__value">${(nli.entailment * 100).toFixed(1)}%</span>
          </div>
          <div class="score-bar__track">
            <div class="score-bar__fill" data-width="${nli.entailment * 100}"></div>
          </div>
        </div>
        <div class="score-bar score-bar--contradiction">
          <div class="score-bar__header">
            <span class="score-bar__label">Contradiction</span>
            <span class="score-bar__value">${(nli.contradiction * 100).toFixed(1)}%</span>
          </div>
          <div class="score-bar__track">
            <div class="score-bar__fill" data-width="${nli.contradiction * 100}"></div>
          </div>
        </div>
        <div class="score-bar score-bar--neutral">
          <div class="score-bar__header">
            <span class="score-bar__label">Neutral</span>
            <span class="score-bar__value">${(nli.neutral * 100).toFixed(1)}%</span>
          </div>
          <div class="score-bar__track">
            <div class="score-bar__fill" data-width="${nli.neutral * 100}"></div>
          </div>
        </div>
        <div class="score-bar score-bar--similarity">
          <div class="score-bar__header">
            <span class="score-bar__label">Best Similarity</span>
            <span class="score-bar__value">${(sim * 100).toFixed(1)}%</span>
          </div>
          <div class="score-bar__track">
            <div class="score-bar__fill" data-width="${sim * 100}"></div>
          </div>
        </div>
      </div>
      <div class="model-card__footer">
        <span class="final-score">Score: <span>${model.final_score?.toFixed(3) || 'N/A'}</span></span>
        <span class="model-card__badge ${model.hallucinated ? 'badge--refuted' : 'badge--supported'}" style="font-size:0.65rem">
          ${model.hallucinated ? '🚨 Hallucinated' : '✅ Factual'}
        </span>
      </div>
    `;

    grid.appendChild(card);
  });

  document.getElementById('resultsSection').classList.remove('hidden');

  // Animate bars after render
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.score-bar__fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    }, 50);
  });
}

function getBadgeClass(label) {
  if (label === 'SUPPORTED') return 'badge--supported';
  if (label === 'REFUTED') return 'badge--refuted';
  return 'badge--nei';
}

function formatModelName(name) {
  const map = {
    'facebook/bart-large-mnli': '🔵 BART Large MNLI',
    'roberta-large-mnli': '🟢 RoBERTa Large MNLI',
    'typeform/distilbert-base-uncased-mnli': '🟣 DistilBERT MNLI'
  };
  return map[name] || name;
}

// ===== EVIDENCE & AGGREGATION TABLES =====
function renderAggregationTable(data) {
  const tbody = document.getElementById('aggregationBody');
  tbody.innerHTML = '';

  data.model_results.forEach(model => {
    const tr = document.createElement('tr');
    const nli = model.nli_scores || {};
    const sim = model.similarity_score || 0;

    const rawHoverStr = JSON.stringify(model.raw_final_result || {}, null, 2);

    tr.innerHTML = `
      <td style="font-weight:600;">${formatModelName(model.model_name)}</td>
      <td class="has-tooltip" data-tooltip="${rawHoverStr}">${model.final_score?.toFixed(3) || '0.000'}</td>
      <td>${(nli.entailment || 0).toFixed(3)}</td>
      <td>${(nli.contradiction || 0).toFixed(3)}</td>
      <td>${sim.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  });

  document.getElementById('aggregationSection').classList.remove('hidden');
}

function renderEvidenceTable(data) {
  const tbody = document.getElementById('evidenceBody');
  tbody.innerHTML = '';

  data.model_results.forEach(model => {
    const evidence = model.evidence || [];
    
    evidence.forEach((ev, i) => {
      const tr = document.createElement('tr');
      const shortTitle = ev.title.length > 25 ? ev.title.substring(0, 25) + '...' : ev.title;
      const shortText = ev.text.length > 80 ? ev.text.substring(0, 80) + '...' : ev.text;
      const ent = ev.nli_scores?.entailment || 0;

      const rawHoverStr = JSON.stringify({
        title: ev.title,
        text: ev.text,
        retriever_score: ev.retriever_score,
        similarity_score: ev.similarity_score,
        nli_scores: ev.nli_scores
      }, null, 2).replace(/"/g, "'");

      tr.innerHTML = `
        <td style="color:var(--text-muted)">${i + 1}</td>
        <td>${formatModelName(model.model_name).split(' ')[1]}</td>
        <td class="has-tooltip" data-tooltip="Title: ${ev.title}">${shortTitle}</td>
        <td class="has-tooltip" data-tooltip="${rawHoverStr}">${shortText}</td>
        <td style="color:var(--accent-cyan);font-family:monospace">${ev.retriever_score.toFixed(3)}</td>
        <td style="color:var(--accent-green);font-family:monospace">${ent.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });
  });

  document.getElementById('evidenceSection').classList.remove('hidden');
}

// ===== TERMINAL TRACE =====
function renderTerminalLog(data) {
  const logEl = document.getElementById('terminalLog');
  let output = '';

  data.model_results.forEach(model => {
    output += `========================================\n`;
    output += `RUNNING WITH MODEL: ${model.model_name}\n`;
    output += `========================================\n\n`;
    
    output += `==============================\n`;
    output += `CLAIM: ${data.claim}\n`;
    output += `==============================\n`;
    output += `Atomic Claims: ['${data.claim}']\n\n`;
    output += `--- Retrieving Evidence ---\n\n`;

    if (model.evidence && model.evidence.length > 0) {
      model.evidence.forEach((ev, i) => {
        output += `Evidence ${i + 1}\n`;
        output += `Title: ${ev.title}\n`;
        const shortText = ev.text.length > 200 ? ev.text.substring(0, 200) + ' ...' : ev.text;
        output += `Text: ${shortText}\n`;
        output += `Retriever Score: ${ev.retriever_score}\n`;
        output += `Similarity Score: ${ev.similarity_score}\n`;
        
        const c = ev.nli_scores.contradiction || 0;
        const n = ev.nli_scores.neutral || 0;
        const e = ev.nli_scores.entailment || 0;
        
        output += `NLI Scores: {'contradiction': ${c}, 'neutral': ${n}, 'entailment': ${e}}\n\n`;
      });
    }

    output += `--- Aggregating Evidence ---\n\n`;

    const raw = model.raw_final_result || {};
    // Emulate Python dict string
    let dictStr = JSON.stringify(raw)
      .replace(/"([^"]+)":/g, "'$1': ")
      .replace(/"/g, "'")
      .replace(/: false/g, ": False")
      .replace(/: true/g, ": True");
    
    output += `Final Decision: ${dictStr}\n\n`;

    output += `--- RESULT ---\n`;
    output += `Claim: ${data.claim}\n`;
    output += `Prediction (${model.model_name}): ${model.label}\n`;
    output += `---------------------------\n\n`;
  });

  logEl.textContent = output;
  document.getElementById('terminalSection').classList.remove('hidden');
}

// ===== EXPERIMENT RESULTS =====
async function loadExperimentResults() {
  try {
    const response = await fetch('/api/results');
    if (!response.ok) return;
    experimentData = await response.json();
    renderExperimentCards(experimentData);
    renderExperimentSelector(experimentData);
    if (Object.keys(experimentData).length > 0) {
      const firstKey = Object.keys(experimentData)[0];
      renderResultsTable(experimentData[firstKey].results);
    }
  } catch (err) {
    console.error('Failed to load experiment results:', err);
  }
}

function renderExperimentCards(data) {
  const grid = document.getElementById('experimentGrid');
  grid.innerHTML = '';

  const configs = Object.entries(data);
  const configLabels = {
    'exp1_baseline': '⚙️ Baseline',
    'exp2_high_recall': '🎯 High Recall',
    'exp3_nli_focused': '🧠 NLI Focused',
    'exp4_strict': '🔒 Strict'
  };

  configs.forEach(([key, val]) => {
    const card = document.createElement('div');
    card.className = 'experiment-card';
    const m = val.metrics;

    const label = configLabels[key] || key;

    card.innerHTML = `
      <div class="experiment-card__name">${label}</div>
      <div class="metric-row">
        <span class="metric-label">Accuracy</span>
        <span class="metric-value ${getMetricClass(m.accuracy)}">${(m.accuracy * 100).toFixed(1)}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Precision</span>
        <span class="metric-value ${getMetricClass(m.precision)}">${(m.precision * 100).toFixed(1)}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Recall</span>
        <span class="metric-value ${getMetricClass(m.recall)}">${(m.recall * 100).toFixed(1)}%</span>
      </div>
      <div class="metric-row">
        <span class="metric-label">F1 Score</span>
        <span class="metric-value ${getMetricClass(m.f1)}">${(m.f1 * 100).toFixed(1)}%</span>
      </div>
    `;

    grid.appendChild(card);
  });
}

function getMetricClass(value) {
  if (value >= 0.7) return 'metric-value--good';
  if (value >= 0.5) return 'metric-value--ok';
  return 'metric-value--bad';
}

function renderExperimentSelector(data) {
  const container = document.getElementById('experimentSelect');
  const configs = Object.keys(data);

  const select = document.createElement('select');
  select.style.cssText = `
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px 16px;
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    outline: none;
    cursor: pointer;
  `;

  configs.forEach(key => {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    select.appendChild(opt);
  });

  select.addEventListener('change', () => {
    renderResultsTable(data[select.value].results);
  });

  container.innerHTML = '';
  container.appendChild(select);
}

function renderResultsTable(results) {
  const tbody = document.getElementById('resultsTableBody');
  tbody.innerHTML = '';

  if (!results) return;

  results.forEach((r, i) => {
    const match = r.true_label === r.predicted_label;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:var(--text-muted)">${i + 1}</td>
      <td>${r.claim}</td>
      <td><span class="model-card__badge ${getBadgeClass(r.true_label)}" style="font-size:0.7rem">${r.true_label}</span></td>
      <td><span class="model-card__badge ${getBadgeClass(r.predicted_label)}" style="font-size:0.7rem">${r.predicted_label}</span></td>
      <td class="${match ? 'correct' : 'incorrect'}" style="font-weight:600">${match ? '✓' : '✗'}</td>
    `;
    tbody.appendChild(tr);
  });
}
