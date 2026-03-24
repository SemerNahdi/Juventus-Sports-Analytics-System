/**
 * Juventus Sports Analytics Dashboard Logic
 */

// Global Chart Options
Chart.defaults.color = '#333333'; // Darker text for Light theme default
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(0, 0, 0, 0.05)';
Chart.defaults.scale.grid.borderColor = 'transparent';

const JUVE_GOLD = '#B38E46'; // Muted gold for Light mode default
const JUVE_BLACK = '#000000';
const JUVE_WHITE = '#FFFFFF';

// Chart instances
let speedChartObj = null;
let riskChartObj = null;
let jointChartObj = null;
let rhythmChartObj = null;
let timingChartObj = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    loadData();
    setupExport();
    setupTheme();
});

function setupTheme() {
    const toggle = document.getElementById('checkbox');
    if (!toggle) return;
    
    toggle.addEventListener('change', (e) => {
        const nextTheme = e.target.checked ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', nextTheme);
        
        // Update Chart font colors for readability
        const isDark = nextTheme === 'dark';
        Chart.defaults.color = isDark ? '#a0a0a0' : '#333';
        Chart.defaults.scale.grid.color = isDark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.05)';
        
        // Dynamic Gold Color
        const goldValue = isDark ? '#C6A15B' : '#B38E46';
        
        // Re-render charts
        if (speedChartObj) speedChartObj.destroy();
        if (riskChartObj) riskChartObj.destroy();
        if (jointChartObj) jointChartObj.destroy();
        if (rhythmChartObj) rhythmChartObj.destroy();
        if (timingChartObj) timingChartObj.destroy();
        loadData();
    });
}

function setupExport() {
    const btn = document.getElementById('exportBtn');
    if (!btn) return;
    btn.addEventListener('click', async () => {
        try {
            const response = await fetch('Output/report.txt');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'Juventus_Analytics_Report.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            alert('Failed to export report: ' + error.message);
        }
    });
}

async function loadData() {
    try {
        const metricsResponse = await fetch('Output/metrics.json');
        if (!metricsResponse.ok) throw new Error('Failed to load Output/metrics.json');
        const data = await metricsResponse.json();
        
        const bioResponse = await fetch('Output/bio.csv');
        let bioStats = null;
        if (bioResponse.ok) {
            const bioText = await bioResponse.text();
            bioStats = parseBioCsv(bioText);
        }
        
        updateSummary(data.player_summary, bioStats);
        renderCharts(data.frame_metrics);
        
    } catch (error) {
        console.error('Core data load error:', error);
    }
}

function parseBioCsv(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length < 2) return null;
    
    const headers = lines[0].split(',');
    const rows = lines.slice(1).map(line => line.split(','));
    
    const getColIndex = (name) => headers.findIndex(h => h.trim() === name);
    
    const calculateAvg = (colName) => {
        const idx = getColIndex(colName);
        if (idx === -1) return 0;
        const validValues = rows.map(r => parseFloat(r[idx])).filter(val => !isNaN(val));
        if (validValues.length === 0) return 0;
        return validValues.reduce((a, b) => a + b, 0) / validValues.length;
    };

    return {
        stepWidth: calculateAvg('step_width'),
        trunkLean: calculateAvg('trunk_lateral_lean'),
        trunkSagittal: calculateAvg('trunk_sagittal_lean'),
        doubleSupport: calculateAvg('double_support') * 100,
        pelvisRot: calculateAvg('pelvis_rotation'),
        armSwing: calculateAvg('left_arm_swing')
    };
}

function updateSummary(summary, bio) {
    if (!summary) return;

    // Professional Standards for Comparison
    const STANDARDS = {
        risk: 25,
        speed: 4.5,
        energy: 550,
        symmetry: 94
    };

    const setTrend = (elementId, current, standard, higherIsBetter = true) => {
        const el = document.getElementById(elementId);
        if (!el) return;
        
        const diff = current - standard;
        const percent = Math.abs((diff / standard) * 100).toFixed(0);
        
        // Determine if result is "good" or "bad"
        const isGood = higherIsBetter ? current >= standard : current <= standard;
        const isUp = current > standard;
        
        el.className = `trend-indicator ${isUp ? 'trend-up' : 'trend-down'} ${isGood ? 'status-good' : 'status-bad'}`;
        el.innerText = `${percent}%`;
    };

    const kpiSet = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    // Risk Card
    kpiSet('kpiRisk', `${summary.peak_risk_score.toFixed(1)}/100`);
    setTrend('riskTrend', summary.peak_risk_score, STANDARDS.risk, false);
    
    const riskCard = document.getElementById('riskCard');
    if (riskCard) {
        riskCard.className = 'kpi-card glass';
        const label = (summary.fall_risk_label || 'low').toLowerCase();
        if (label === 'high') riskCard.classList.add('risk-high');
        else if (label === 'medium') riskCard.classList.add('risk-medium');
        else riskCard.classList.add('risk-low');
        kpiSet('kpiRiskLabel', `Overall Risk: ${summary.fall_risk_label || 'Low'}`);
    }

    // Speed Card
    kpiSet('kpiSpeed', `${summary.max_speed.toFixed(2)} m/s`);
    setTrend('speedTrend', summary.max_speed, STANDARDS.speed, true);
    kpiSet('kpiSpeedLabel', `Avg: ${summary.avg_speed.toFixed(2)} m/s`);

    // Energy Card
    kpiSet('kpiEnergy', `${summary.estimated_energy_kcal_hr.toFixed(0)} kcal/hr`);
    setTrend('energyTrend', summary.estimated_energy_kcal_hr, STANDARDS.energy, true);
    kpiSet('kpiDistance', `Total Stride: ${summary.avg_stride_length.toFixed(2)}m`);

    // Symmetry Card
    kpiSet('kpiSymmetry', `${summary.gait_symmetry_pct.toFixed(1)}%`);
    setTrend('symmetryTrend', summary.gait_symmetry_pct, STANDARDS.symmetry, true);
    kpiSet('kpiCadence', `Cadence: ${summary.avg_cadence.toFixed(0)} spm`);

    if (bio) {
        kpiSet('valStepWidth', `${bio.stepWidth.toFixed(2)} m`);
        kpiSet('valTrunkLean', `${Math.abs(bio.trunkLean).toFixed(1)}°`);
        kpiSet('valDoubleSupport', `${bio.doubleSupport.toFixed(1)}%`);
        kpiSet('valPelvicRot', `${Math.abs(bio.pelvicRot).toFixed(1)}°`);
        kpiSet('valTrunkSag', `${Math.abs(bio.trunkSagittal).toFixed(1)}°`);
        kpiSet('valArmSwing', `${bio.armSwing.toFixed(1)}°`);
    }
}

function renderCharts(frames) {
    if (!frames || frames.length === 0) return;

    const step = Math.max(1, Math.floor(frames.length / 150));
    const sampledFrames = frames.filter((_, i) => i % step === 0);
    const labels = sampledFrames.map(f => f.timestamp.toFixed(2) + 's');
    
    const pointHover = { pointRadius: 0, pointHoverRadius: 6 };

    // Speed & Acceleration - Muted harmonious colors
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    speedChartObj = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Speed (m/s)',
                    data: sampledFrames.map(f => f.speed),
                    borderColor: JUVE_GOLD,
                    backgroundColor: 'rgba(198, 161, 91, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    ...pointHover,
                    yAxisID: 'y'
                },
                {
                    label: 'Acceleration (m/s²)',
                    data: sampledFrames.map(f => f.acceleration),
                    borderColor: '#6b7280', // Slate Gray
                    borderDash: [5, 5],
                    borderWidth: 2,
                    tension: 0.4,
                    fill: false,
                    ...pointHover,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { beginAtZero: true, title: { display: true, text: 'SPEED' } },
                y1: { position: 'right', display: true, title: { display: true, text: 'ACCEL' }, grid: { drawOnChartArea: false } }
            }
        }
    });

    // Risk Analysis - All fills
    const riskCtx = document.getElementById('riskChart').getContext('2d');
    riskChartObj = new Chart(riskCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Overall Risk Score',
                    data: sampledFrames.map(f => f.risk_score),
                    borderColor: '#ef4444', 
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    ...pointHover
                },
                {
                    label: 'Joint Stress',
                    data: sampledFrames.map(f => f.joint_stress * 100),
                    borderColor: JUVE_GOLD,
                    backgroundColor: 'rgba(198, 161, 91, 0.15)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true,
                    ...pointHover
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { ticks: { maxTicksLimit: 10 } },
                y: { max: 100, beginAtZero: true, title: { display: true, text: 'PERCENTAGE/SCORE' } }
            }
        }
    });

    // Calculate Global Averages for Bar-only charts
    const avg = (arr) => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
    const avgStride = avg(frames.map(f => f.stride_length));
    const avgCadence = avg(frames.map(f => f.cadence));
    const avgStepTime = avg(frames.map(f => f.step_time));
    const avgFlightTime = avg(frames.map(f => f.flight_time));

    // Rhythm Chart (Stride vs Cadence) - Average Summary
    const rhythmCtx = document.getElementById('rhythmChart').getContext('2d');
    rhythmChartObj = new Chart(rhythmCtx, {
        type: 'bar',
        data: {
            labels: ['Cadence (spm)', 'Stride (m)'],
            datasets: [
                {
                    label: 'Cadence (spm)',
                    data: [avgCadence, 0],
                    backgroundColor: '#f59e0b',
                    borderRadius: 8,
                    yAxisID: 'y'
                },
                {
                    label: 'Stride Length (m)',
                    data: [0, avgStride],
                    backgroundColor: JUVE_GOLD,
                    borderRadius: 8,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: { 
                    callbacks: { 
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}` 
                    } 
                }
            },
            scales: {
                y: { 
                    beginAtZero: true, 
                    title: { display: true, text: 'SPM' } 
                },
                y1: {
                    position: 'right',
                    beginAtZero: true,
                    title: { display: true, text: 'METERS' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });

    // Timing Chart (Step vs Flight) - Average Summary
    const timingCtx = document.getElementById('timingChart').getContext('2d');
    timingChartObj = new Chart(timingCtx, {
        type: 'bar',
        data: {
            labels: ['Step Time (s)', 'Flight Time (s)'],
            datasets: [
                {
                    label: 'Session Average',
                    data: [avgStepTime, avgFlightTime],
                    backgroundColor: ['#3b82f6', '#10b981'],
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(3)}s` } }
            },
            scales: {
                y: { 
                    beginAtZero: true, 
                    title: { display: true, text: 'Seconds' }
                }
            }
        }
    });

    // Knee Flexion
    const jointCtx = document.getElementById('jointChart').getContext('2d');
    jointChartObj = new Chart(jointCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Left Knee',
                    data: sampledFrames.map(f => f.left_knee_angle),
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    tension: 0.4,
                    ...pointHover
                },
                {
                    label: 'Right Knee',
                    data: sampledFrames.map(f => f.right_knee_angle),
                    borderColor: '#9ca3af',
                    borderWidth: 2,
                    tension: 0.4,
                    ...pointHover
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { ticks: { maxTicksLimit: 15 } },
                y: { title: { display: true, text: 'DEGREES' } }
            }
        }
    });
}
