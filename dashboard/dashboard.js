/**
 * Sports Analytics System Dashboard Logic
 */

// Global Chart Options
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';

// Chart instances
let speedChartObj = null;
let riskChartObj = null;
let jointChartObj = null;
let valgusChartObj = null;
let trunkChartObj = null;

// API Base URL (adjust if running on different port)
const API_BASE = window.location.origin;

let jobPollInterval = null;
let jobPollLogIndex = 0;
let activePollingJobId = null;
let currentJobId = null;

const IN_PROGRESS_STATUSES = new Set(['processing', 'pending', 'cancelling']);

function isJobInProgress(status) {
    return IN_PROGRESS_STATUSES.has((status || '').toLowerCase());
}

function toNumber(value, fallback = 0) {
    if (value === null || value === undefined || value === '') return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function asArray(value) {
    return Array.isArray(value) ? value : [];
}

function parseMaybeJson(value, fallback) {
    if (value && typeof value === 'object') return value;
    if (typeof value !== 'string') return fallback;
    try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === 'object' ? parsed : fallback;
    } catch (_error) {
        return fallback;
    }
}

function getAnalysisEnvelope(analysis) {
    const summary = parseMaybeJson(analysis?.summary, {});
    const metadata = parseMaybeJson(analysis?.metadata || summary.metadata, {});
    const playerSummary = parseMaybeJson(summary.player_summary || analysis?.player_summary, {});
    const biomechanicsSummary = parseMaybeJson(analysis?.biomechanics_summary || summary.biomechanics_summary, {});
    const frameMetrics = asArray(parseMaybeJson(summary.frame_metrics || analysis?.frame_metrics || analysis?.frames, []));
    const dataUrls = parseMaybeJson(analysis?.data_urls || summary.data_urls, {});
    const plotUrls = parseMaybeJson(analysis?.plot_urls || summary.plot_urls, {});
    const sports2dFiles = parseMaybeJson(analysis?.sports2d_output_files || summary.sports2d_output_files, {});

    return {
        summary,
        metadata,
        playerSummary,
        biomechanicsSummary,
        frameMetrics,
        dataUrls,
        plotUrls,
        sports2dFiles,
    };
}

function getMetricAverage(frames, key) {
    const values = frames
        .map(frame => toNumber(frame?.[key], NaN))
        .filter(value => Number.isFinite(value));
    if (!values.length) return 0;
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function getMetricAverageFromKeys(frames, keys) {
    const values = frames
        .map(frame => {
            for (const key of keys) {
                const value = toNumber(frame?.[key], NaN);
                if (Number.isFinite(value)) return value;
            }
            return NaN;
        })
        .filter(value => Number.isFinite(value));
    if (!values.length) return 0;
    return values.reduce((total, value) => total + value, 0) / values.length;
}

function getSummaryValue(summary, keys, fallback = 0) {
    for (const key of keys) {
        const value = toNumber(summary?.[key], NaN);
        if (Number.isFinite(value)) return value;
    }
    return fallback;
}

// Toast Utility
function showToast(title, message, icon = 'fa-check') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <div class="toast-icon"><i class="fa-solid ${icon}"></i></div>
        <div class="toast-content">
            <h4>${title}</h4>
            <p>${message}</p>
        </div>
    `;
    container.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Auto-remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 5000);
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const jobId = urlParams.get('job_id');
    
    if (jobId) {
        loadByJobId(jobId);
    } else {
        loadLatest();
    }
});

// --- Navigation & UI ---

async function toggleHistory() {
    const section = document.getElementById('historySection');
    if (section.style.display === 'none') {
        section.style.display = 'block';
        await fetchHistory();
    } else {
        section.style.display = 'none';
    }
}

function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('collapsed');
}

// --- Data Fetching ---

/**
 * Cache Strategy: 
 * 1. Check localStorage first
 * 2. Display cached data immediately
 * 3. Fetch from API in background to ensure data is fresh
 */

const CACHE_KEY_PREFIX = 'mitus_analysis_';
const CACHE_HISTORY_KEY = 'mitus_history';

function getCachedAnalysis(jobId) {
    const cached = localStorage.getItem(CACHE_KEY_PREFIX + jobId);
    return cached ? JSON.parse(cached) : null;
}

function setCachedAnalysis(analysis) {
    if (analysis && analysis.id) {
        localStorage.setItem(CACHE_KEY_PREFIX + analysis.id, JSON.stringify(analysis));
        localStorage.setItem('mitus_latest_job_id', analysis.id);
    }
}

function showProcessingPanel(analysis) {
    const section = document.getElementById('jobProcessingSection');
    const grid = document.getElementById('dashboardGrid');
    if (section) section.style.display = 'block';
    if (grid) grid.style.display = 'none';

    const title = document.getElementById('jobProcessingTitle');
    const statusEl = document.getElementById('jobProcessingStatus');
    if (title) title.textContent = 'Analysis in progress';
    if (statusEl) {
        statusEl.textContent = 'Your results will appear here automatically when processing finishes.';
        statusEl.style.color = '';
    }
    updateProcessingTelemetry(analysis);
}

function hideProcessingPanel() {
    const section = document.getElementById('jobProcessingSection');
    const grid = document.getElementById('dashboardGrid');
    if (section) section.style.display = 'none';
    if (grid) grid.style.display = '';
}

function updateProcessingTelemetry(analysis) {
    const logs = asArray(analysis?.logs);
    const statusEl = document.getElementById('jobProcessingStatus');
    const bar = document.getElementById('jobProcessingBar');
    const logBox = document.getElementById('jobProcessingLogs');

    for (let i = jobPollLogIndex; i < logs.length; i++) {
        const raw = logs[i];
        const msg = typeof raw === 'string' ? (raw.split(' - ').pop() || raw) : String(raw);
        if (statusEl) statusEl.textContent = msg;
        if (logBox) {
            const line = document.createElement('div');
            line.style.padding = '4px 0';
            line.textContent = msg;
            logBox.prepend(line);
            while (logBox.children.length > 8) logBox.lastElementChild.remove();
        }
        jobPollLogIndex++;
    }

    if (bar) {
        let pct = Math.min(92, 12 + jobPollLogIndex * 6);
        if (logs.some(l => String(l).toLowerCase().includes('finalized'))) pct = 98;
        bar.style.width = `${pct}%`;
    }
}

function stopJobPolling() {
    if (jobPollInterval) {
        clearInterval(jobPollInterval);
        jobPollInterval = null;
    }
    activePollingJobId = null;
}

function startJobPolling(jobId) {
    stopJobPolling();
    activePollingJobId = jobId;
    jobPollLogIndex = 0;
    localStorage.setItem('mitus_active_job_id', jobId);

    const cancelBtn = document.getElementById('dashboardCancelBtn');
    if (cancelBtn) {
        cancelBtn.onclick = async () => {
            if (!confirm('Stop this analysis?')) return;
            cancelBtn.disabled = true;
            try {
                await fetch(`${API_BASE}/analyses/${jobId}/cancel`, { method: 'POST' });
            } catch (e) {
                console.warn('Cancel request failed', e);
            }
        };
    }

    jobPollInterval = setInterval(() => pollJobUntilDone(jobId), 2500);
    pollJobUntilDone(jobId);
}

async function pollJobUntilDone(jobId) {
    try {
        const response = await fetch(`${API_BASE}/analyses/${jobId}`);
        if (!response.ok) return;

        const analysis = await response.json();
        setCachedAnalysis(analysis);

        if (isJobInProgress(analysis.status)) {
            showProcessingPanel(analysis);
            return;
        }

        stopJobPolling();
        localStorage.removeItem('mitus_active_job_id');

        if (analysis.status === 'success') {
            hideProcessingPanel();
            displayAnalysis(analysis);
            showToast('Analysis complete', 'Results are ready.', 'fa-circle-check');
            return;
        }

        const title = document.getElementById('jobProcessingTitle');
        const statusEl = document.getElementById('jobProcessingStatus');
        if (title) title.textContent = analysis.status === 'cancelled' ? 'Analysis cancelled' : 'Analysis failed';
        if (statusEl) {
            statusEl.textContent = analysis.error || 'The job did not complete successfully.';
            statusEl.style.color = '#f87171';
        }
    } catch (error) {
        console.error('Job polling error:', error);
    }
}

function routeAnalysisView(analysis, jobId) {
    if (!analysis) return;
    if (jobId || analysis.id) {
        currentJobId = jobId || analysis.id;
    }
    if (isJobInProgress(analysis.status)) {
        showProcessingPanel(analysis);
        startJobPolling(jobId || analysis.id);
        return;
    }
    hideProcessingPanel();
    stopJobPolling();
    localStorage.removeItem('mitus_active_job_id');
    displayAnalysis(analysis);
}

async function loadByJobId(jobId) {
    const cached = getCachedAnalysis(jobId);
    if (cached) {
        routeAnalysisView(cached, jobId);
    }

    try {
        const response = await fetch(`${API_BASE}/analyses/${jobId}`);
        if (response.ok) {
            const analysis = await response.json();
            setCachedAnalysis(analysis);
            routeAnalysisView(analysis, jobId);
        } else if (!cached) {
            console.error(`Analysis ${jobId} not found.`);
            loadLatest();
        }
    } catch (error) {
        console.error('Error fetching analysis:', error);
        if (!cached) loadLatest();
    }
}

async function loadLatest() {
    const latestId = localStorage.getItem('mitus_latest_job_id');
    if (latestId) {
        await loadByJobId(latestId);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/analyses/latest`);
        if (response.ok) {
            const analysis = await response.json();
            setCachedAnalysis(analysis);
            routeAnalysisView(analysis, analysis.id);
        } else {
            console.log("No previous analyses found.");
        }
    } catch (error) {
        console.error('Error fetching latest:', error);
    }
}

async function fetchHistory() {
    const list = document.getElementById('historyList');
    
    // 1. Load history from cache if available
    const cachedHistory = localStorage.getItem(CACHE_HISTORY_KEY);
    if (cachedHistory) {
        renderHistory(JSON.parse(cachedHistory));
    }

    // 2. Fetch fresh history
    try {
        const response = await fetch(`${API_BASE}/analyses`);
        const data = await response.json();
        localStorage.setItem(CACHE_HISTORY_KEY, JSON.stringify(data));
        renderHistory(data);
    } catch (error) {
        if (!cachedHistory) {
            list.innerHTML = '<p class="meta">Error loading history</p>';
        }
    }
}

function renderHistory(data) {
    const list = document.getElementById('historyList');
    list.innerHTML = '';
    if (data.length === 0) {
        list.innerHTML = '<p class="meta">No history found</p>';
        return;
    }

    data.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        const date = new Date(item.created_at).toLocaleString();
        div.innerHTML = `
            <h4>Player #${item.player_id} - ${item.session_tags || 'Quick Scan'}</h4>
            <div class="meta">${date} | ${item.yolo_size.toUpperCase()} Tracking</div>
        `;
        div.onclick = () => {
            displayAnalysis(item);
            toggleHistory();
        };
        list.appendChild(div);
    });
}

function displayAnalysis(analysis) {
    if (!analysis) return;
    if (isJobInProgress(analysis.status)) {
        routeAnalysisView(analysis, analysis.id);
        return;
    }
    hideProcessingPanel();
    if (analysis.id) {
        currentJobId = analysis.id;
    }
    const envelope = getAnalysisEnvelope(analysis);

    // Update Header
    const playerId = envelope.playerSummary.player_id || analysis.player_id || '--';
    const backendLabel = envelope.metadata.angle_backend || envelope.metadata.pipeline || 'unified';
    const totalFrames = envelope.metadata.total_frames || envelope.playerSummary.total_frames || envelope.frameMetrics.length || 0;
    const sessionLabel = analysis.session_tags || analysis.created_at || envelope.metadata.video_path || envelope.metadata.video_source || 'Local output';
    const sessionStamp = analysis.created_at ? new Date(analysis.created_at).toLocaleDateString() : sessionLabel;
    document.getElementById('viewTitle').textContent = `Analysis: Player #${playerId}`;
    document.getElementById('sessionInfo').textContent = `${totalFrames} frames | ${backendLabel} | ${sessionStamp}`;

    // Update Video (Adopted from Old DASH / Native compatibility)
    const video = document.getElementById('analysisVideo');
    
    // Add error listener to help debug codec issues
    video.onerror = () => {
        const err = video.error;
        console.error("Video Playback Error:", err);
        if (err && err.code === 4) {
            alert("Video Codec Error: Your browser cannot play this video format. Try a different browser like Chrome or Edge.");
        }
    };

    if (analysis.video_url) {
        console.log("Applying video source:", analysis.video_url);
        
        // Reset video state
        video.pause();
        
        // Update both the video src and the source element for maximum compatibility
        const source = video.querySelector('source');
        if (source) {
            source.src = analysis.video_url;
            source.type = 'video/mp4'; 
        }
        video.src = analysis.video_url; 
        
        // Ensure muted autoplay (standard for dashboard visuals)
        video.muted = true; 
        video.load();
        
        // Slight delay for source internal buffering
        setTimeout(() => {
            const playPromise = video.play();
            if (playPromise !== undefined) {
                playPromise.catch(e => {
                    console.warn("Auto-play blocked or failed:", e);
                    // If it failed because of user interaction required, it's okay
                });
            }
        }, 150);
    }

    // --- NEW: Sync URL with the ID so users can share direct links ---
    if (analysis.id) {
        const url = new URL(window.location);
        url.searchParams.set('job_id', analysis.id);
        window.history.replaceState({}, '', url);
    }

    // --- NEW: Email Notification Toast ---
    if (analysis.email && analysis.status === 'success') {
        showToast("Report Sent", `Finalized findings emailed to ${analysis.email}`, 'fa-envelope-circle-check');
    }

    // Update KPIs and Charts
    updateSummary(analysis);
    updateMATSummary(analysis);
    
    // Update Resources
    populateResources(analysis);
    
    if (envelope.frameMetrics.length > 0) {
        renderCharts(envelope.frameMetrics);
    } else if (envelope.dataUrls && envelope.dataUrls['analytics_unified.json']) {
        // Fallback: fetch from public URL
        fetch(envelope.dataUrls['analytics_unified.json'])
            .then(res => res.json())
            .then(data => renderCharts(data.frame_metrics || data.frames || []));
    }
}

function populateResources(analysis) {
    const dataList = document.getElementById('dataFilesList');
    const plotList = document.getElementById('plotFilesList');
    const sports2dList = document.getElementById('sports2dFilesList');
    const metaList = document.getElementById('analysisMetaList');
    const envelope = getAnalysisEnvelope(analysis);

    dataList.innerHTML = '';
    plotList.innerHTML = '';
    if (sports2dList) sports2dList.innerHTML = '';
    if (metaList) metaList.innerHTML = '';

    if (metaList) {
        const metadataItems = [
            ['Player ID', envelope.playerSummary.player_id || analysis.player_id || '--', 'ID'],
            ['Frames', envelope.metadata.total_frames || envelope.playerSummary.total_frames || envelope.frameMetrics.length || 0, 'FRM'],
            ['Backend', envelope.metadata.angle_backend || 'unified', 'PIPE'],
            ['Source', envelope.metadata.video_path ? (String(envelope.metadata.video_path).split(/[\\/]/).pop() || envelope.metadata.video_path) : 'session', 'SRC'],
        ];

        metadataItems.forEach(([name, value, type]) => {
            metaList.appendChild(createFileLink(`${name}: ${value}`, null, type));
        });
    }

    // Data Files
    if (envelope.dataUrls) {
        Object.entries(envelope.dataUrls).forEach(([name, url]) => {
            const ext = name.split('.').pop().toUpperCase();
            dataList.appendChild(createFileLink(name, url, ext));
        });
    }

    // Static Plots
    if (envelope.plotUrls) {
        Object.entries(envelope.plotUrls).forEach(([name, url]) => {
            const ext = name.split('.').pop().toUpperCase();
            plotList.appendChild(createFileLink(name, url, ext));
        });
    }

    if (sports2dList) {
        const sports2dEntries = Object.entries(envelope.sports2dFiles || {});
        if (sports2dEntries.length === 0) {
            sports2dList.appendChild(createFileLink('No native Sports2D files', null, 'N/A'));
        } else {
            sports2dEntries.forEach(([name, url]) => {
                const ext = name.split('.').pop().toUpperCase();
                sports2dList.appendChild(createFileLink(name, url, ext));
            });
        }
    }
}

function createFileLink(name, url, type) {
    const el = document.createElement(url ? 'a' : 'div');
    if (url) {
        el.href = url;
        el.target = '_blank';
        el.rel = 'noreferrer';
    }
    el.className = url ? 'file-link' : 'file-link file-link-static';
    el.innerHTML = `
        <span>${name}</span>
        <span class="badge type-icon">${type}</span>
    `;
    return el;
}

// --- Upload & Analysis ---


// --- Chart Rendering (adapted from original) ---

function updateSummary(analysis) {
    const envelope = getAnalysisEnvelope(analysis);
    const s = envelope.playerSummary;
    if (!s) return;

    // Peak Risk
    const peakRisk = toNumber(s.peak_risk_score, 0).toFixed(1);
    const riskLabel = s.fall_risk_label || s.injury_risk_label || 'Low';
    document.getElementById('kpiRisk').textContent = `${peakRisk}/100`;
    document.getElementById('kpiRiskLabel').textContent = `Overall Risk: ${riskLabel}`;
    
    const riskCard = document.getElementById('riskCard');
    riskCard.className = 'kpi-card glass';
    if (riskLabel.toLowerCase() === 'high') riskCard.classList.add('risk-high');
    else if (riskLabel.toLowerCase() === 'medium') riskCard.classList.add('risk-medium');
    else riskCard.classList.add('risk-low');

    // Fatigue
    document.getElementById('kpiFatigue').textContent = s.fatigue_label || 'Low';
    const fCard = document.getElementById('fatigueCard');
    fCard.className = 'kpi-card glass'; // Reset
    if (s.fatigue_label === 'High') fCard.classList.add('risk-high');
    else if (s.fatigue_label === 'Medium') fCard.classList.add('risk-medium');
    else fCard.classList.add('risk-low');

    // Injury
    document.getElementById('kpiInjury').textContent = s.injury_risk_label || 'Normal';
    document.getElementById('kpiInjuryDetail').textContent = s.injury_risk_detail || 'No anomalies';
    const iCard = document.getElementById('injuryCard');
    iCard.className = 'kpi-card glass'; // Reset
    if (s.injury_risk_label === 'High') iCard.classList.add('risk-high');
    else iCard.classList.add('risk-low');

    document.getElementById('kpiSpeed').textContent = `${toNumber(s.max_speed, 0).toFixed(2)} m/s`;
    document.getElementById('kpiSpeedLabel').textContent = `Avg: ${toNumber(s.avg_speed, 0).toFixed(2)} m/s`;

    // ── Gait Metrics Rendering ─────────────────────────────────────────────────
    // NOTE: setVal runs unconditionally — existing cards use bioSummary as primary,
    // new gait cards use playerSummary directly. Neither requires frameMetrics to be loaded.
    const metrics = envelope.frameMetrics;
    const bioSummary = envelope.biomechanicsSummary || {};

    console.log('[Dashboard] updateSummary: playerSummary =', s);
    console.log('[Dashboard] updateSummary: biomechanicsSummary =', bioSummary);
    console.log('[Dashboard] updateSummary: frameMetrics length =', metrics.length);

    const setVal = (id, val, unit = '', precision = null) => {
        const el = document.getElementById(id);
        if (!el) {
            console.warn(`[Dashboard] setVal: element #${id} NOT FOUND in DOM`);
            return;
        }
        // Default precision: 2 for width/stride/time, 1 for angles etc.
        const p = precision !== null ? precision :
                 (id.toLowerCase().includes('width') || id.toLowerCase().includes('stride') || id.toLowerCase().includes('time') ? 2 : 1);
        const num = toNumber(val, null);
        if (num === null) {
            // val was non-numeric — keep placeholder
            console.log(`[Dashboard] setVal: #${id} — value missing/non-numeric (raw: ${val}), keeping placeholder`);
            return;
        }
        el.textContent = `${num.toFixed(p)}${unit}`;
    };

    try {
        // Biometric detail cards — bioSummary is primary, frame averages are fallback only
        setVal('valStepWidth',     getSummaryValue(bioSummary, ['step_width_mean'],          toNumber(s.avg_stride_length, 0) * 0.12), ' m');
        setVal('valTrunkLean',     Math.abs(getSummaryValue(bioSummary, ['trunk_lateral_lean_mean'],  getMetricAverageFromKeys(metrics, ['trunk_lateral_lean', 'trunk_lean', 'bio_trunk_lateral_lean']))), '°');
        setVal('valDoubleSupport', getSummaryValue(bioSummary, ['double_support_pct'],        toNumber(s.double_support_pct, 0)), '%');
        setVal('valPelvicRot',     Math.abs(getSummaryValue(bioSummary, ['pelvis_rotation_mean'],     toNumber(s.avg_pelvic_rotation, 0))), '°');
        setVal('valTrunkSag',      Math.abs(getSummaryValue(bioSummary, ['trunk_sagittal_lean_mean'], getMetricAverage(metrics, 'bio_trunk_sagittal_lean'))), '°');
        setVal('valArmSwing',      getSummaryValue(bioSummary, ['arm_swing_asymmetry_mean', 'left_arm_swing_mean', 'right_arm_swing_mean'], getMetricAverage(metrics, 'bio_arm_swing_asymmetry')), '°');

        // Gait metrics — read from playerSummary directly (multiple key aliases for resilience)
        const strideLen   = s.avg_stride_length   ?? s.stride_length   ?? s.avgStrideLength;
        const stepTime    = s.avg_step_time       ?? s.step_time       ?? s.avgStepTime;
        const cadence     = s.avg_cadence         ?? s.cadence         ?? s.avgCadence;
        const flightTime  = s.avg_flight_time     ?? s.flight_time     ?? s.avgFlightTime;

        console.log('[Dashboard] Gait raw values — stride:', strideLen, 'stepTime:', stepTime, 'cadence:', cadence, 'flightTime:', flightTime);

        setVal('valStrideLength', strideLen,   ' m');
        setVal('valStepTime',     stepTime,    ' s');
        setVal('valCadence',      cadence,     ' bpm', 1);
        setVal('valFlightTime',   flightTime,  ' s');
    } catch (gaitErr) {
        console.error('[Dashboard] Error in gait setVal block:', gaitErr);
    }

    try {
        const energyFallback = getMetricAverage(metrics, 'energy_expenditure');
        const energySummary = toNumber(s.estimated_energy_kcal_hr, NaN);
        const energyValue = Number.isFinite(energySummary) && energySummary > 0 ? energySummary : energyFallback;
        document.getElementById('kpiEnergy').textContent = `${energyValue.toFixed(0)} kcal/hr`;
        document.getElementById('kpiDistance').textContent = `Dist: ${toNumber(s.total_distance_m, 0).toFixed(1)} m`;
        document.getElementById('kpiSymmetry').textContent = `${toNumber(s.gait_symmetry_pct, 0).toFixed(1)} %`;
        document.getElementById('kpiStride').textContent = `Stride: ${toNumber(s.avg_stride_length, 0).toFixed(2)} m`;
    } catch (kpiErr) {
        console.error('[Dashboard] Error in KPI bottom block:', kpiErr);
    }
}

function updateMATSummary(analysis) {
    if (!analysis) return;
    const envelope = getAnalysisEnvelope(analysis);
    
    // Look for mat_summary in multiple potential locations
    const mat = analysis.mat_summary || envelope.summary.mat_summary || analysis.summary?.mat_summary;
    console.log('[Dashboard] updateMATSummary: mat_summary found =', !!mat);
    
    const matSection = document.getElementById('matSection');
    const matDivider = document.getElementById('matDivider');
    const matContent = document.getElementById('matContent');
    const matNoData = document.getElementById('matNoData');
    
    if (matSection) matSection.style.display = 'block';
    if (matDivider) matDivider.style.display = 'flex';

    if (!mat || !mat.events || mat.events.length === 0) {
        console.log('[Dashboard] No MAT events found in data.');
        if (matContent) matContent.style.display = 'none';
        if (matNoData) matNoData.style.display = 'block';
        document.getElementById('matProtocolName').textContent = 'NONE';
        document.getElementById('matSymmetryBadge').textContent = 'LSI: N/A';
        return;
    }

    console.log('[Dashboard] Displaying MAT results for protocol:', mat.protocol_id);
    if (matContent) matContent.style.display = 'block';
    if (matNoData) matNoData.style.display = 'none';
    const event = mat.events[0];
    
    document.getElementById('matProtocolName').textContent = (mat.protocol_id || 'Unknown').replace(/_/g, ' ').toUpperCase();
    const lsiScore = toNumber(mat.limb_symmetry_index, 100);
    const lsiBadge = document.getElementById('matSymmetryBadge');
    lsiBadge.textContent = `LSI: ${lsiScore.toFixed(1)}%`;
    
    // Color code LSI: < 85% is a clinical red flag
    lsiBadge.style.background = lsiScore < 85 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)';
    lsiBadge.style.color = lsiScore < 85 ? 'var(--risk-high)' : 'var(--risk-low)';
    lsiBadge.style.borderColor = lsiScore < 85 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)';
    
    const setMATVal = (id, val, unit, threshold = null, isInverse = false) => {
        const el = document.getElementById(id);
        const statusEl = document.getElementById(id + 'Status');
        if (!el) return;
        const num = toNumber(val, 0);
        el.textContent = `${num.toFixed(1)}${unit}`;
        
        if (statusEl && threshold !== null) {
            statusEl.className = 'status-indicator';
            const isWarning = isInverse ? (num < threshold) : (Math.abs(num) > threshold);
            if (isWarning) {
                statusEl.classList.add('status-warning');
            } else {
                statusEl.classList.add('status-success');
            }
        }
    };

    setMATVal('matValgus', event.landing_valgus_left, '°', 10);
    setMATVal('matFlexion', event.peak_knee_flexion_landing, '°', 150, true); // Stiff if > 150 (meaning < 30 deg bend)
    setMATVal('matFlight', event.flight_time, 's');
    setMATVal('matStabilization', event.time_to_stabilization, 's');
    setMATVal('matDistance', event.hop_distance_m, 'm');
}

function renderCharts(frames) {
    if (!frames || frames.length === 0) return;

    // Destroy old charts to prevent overlapping
    if (speedChartObj) speedChartObj.destroy();
    if (riskChartObj) riskChartObj.destroy();
    if (jointChartObj) jointChartObj.destroy();
    if (valgusChartObj) valgusChartObj.destroy();
    if (trunkChartObj) trunkChartObj.destroy();

    const step = Math.max(1, Math.floor(frames.length / 150));
    const sampledFrames = frames.filter((_, i) => i % step === 0);
    const labels = sampledFrames.map(f => `${toNumber(f.timestamp, 0).toFixed(2)}s`);
    
    // --- Speed Chart ---
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    const speedGradient = speedCtx.createLinearGradient(0, 0, 0, 400);
    speedGradient.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    speedGradient.addColorStop(1, 'rgba(0, 240, 255, 0)');

    speedChartObj = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Speed (m/s)',
                data: sampledFrames.map(f => toNumber(f.speed, 0)),
                borderColor: '#00f0ff',
                backgroundColor: speedGradient,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
    });

    // --- Valgus Chart (New) ---
    const valgusCtx = document.getElementById('valgusChart').getContext('2d');
    valgusChartObj = new Chart(valgusCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'L. Valgus',
                data: sampledFrames.map(f => toNumber(f.l_valgus_clinical, 0)),
                borderColor: '#ef4444',
                tension: 0.4,
                pointRadius: 0
            }, {
                label: 'R. Valgus',
                data: sampledFrames.map(f => f.r_valgus_clinical),
                borderColor: '#fbbf24',
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // --- Risk Chart ---
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    riskChartObj = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Score (%)',
                data: sampledFrames.map(f => toNumber(f.risk_score, 0)),
                borderColor: '#ff00ff',
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
    });

    // --- Joint Chart ---
    const jointCtx = document.getElementById('jointChart').getContext('2d');
    jointChartObj = new Chart(jointCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'L. Knee Angle',
                data: sampledFrames.map(f => toNumber(f.left_knee_angle, 0)),
                borderColor: '#10b981',
                borderWidth: 2,
                pointRadius: 0
            }, {
                label: 'R. Knee Angle',
            data: sampledFrames.map(f => toNumber(f.right_knee_angle, 0)),
                borderColor: '#8b5cf6',
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // --- Trunk Chart (New) ---
    const trunkCtx = document.getElementById('trunkChart').getContext('2d');
    trunkChartObj = new Chart(trunkCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Trunk Lean (Deg)',
                data: sampledFrames.map(f => toNumber(f.trunk_lateral_lean ?? f.trunk_lean ?? f.bio_trunk_lateral_lean, 0)),
                borderColor: '#60a5fa',
                backgroundColor: 'rgba(96, 165, 250, 0.2)',
                fill: true,
                tension: 0.1,
                pointRadius: 0
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });
}

async function shareResultsByEmail() {
    const email = document.getElementById('shareEmailInput').value;
    const btn = document.getElementById('shareEmailBtn');
    
    if (!email || !email.includes('@')) {
        alert("Please enter a valid email address.");
        return;
    }

    const jobId = currentJobId; // Assumes currentJobId is tracked globally
    if (!jobId) {
        alert("No active analysis to share.");
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending...';

    try {
        const res = await fetch(`/analyses/${jobId}/email?email=${encodeURIComponent(email)}`, {
            method: 'POST'
        });
        const data = await res.json();
        
        if (res.ok) {
            alert(`Results successfully sent to ${email}`);
            document.getElementById('shareEmailInput').value = '';
        } else {
            alert(`Error: ${data.message || 'Failed to send email'}`);
        }
    } catch (err) {
        alert(`Network error: ${err.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="paper-plane"></i> Send';
    }
}
