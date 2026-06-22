/* =================================================================
   Football Highlight Detection — 3D Frontend Application
   Three.js animated background + full SPA logic
   ================================================================= */

// =====================================================
// 1. THREE.JS 3D BACKGROUND SCENE
// =====================================================
(function initThreeScene() {
    const container = document.getElementById('three-bg');
    if (!container || typeof THREE === 'undefined') return;

    const W = window.innerWidth, H = window.innerHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // ---- Particles ----
    const PARTICLE_COUNT = 2500;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(PARTICLE_COUNT * 3);
    const pCol = new Float32Array(PARTICLE_COUNT * 3);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const r = 15 + Math.random() * 50;
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        pPos[i * 3] = r * Math.sin(ph) * Math.cos(th);
        pPos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
        pPos[i * 3 + 2] = r * Math.cos(ph);
        const t = Math.random();
        pCol[i * 3] = 0.0 + t * 0.48;
        pCol[i * 3 + 1] = 0.3 + t * 0.53;
        pCol[i * 3 + 2] = 0.85 + t * 0.15;
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
    const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({
        size: 0.14, vertexColors: true, transparent: true, opacity: 0.55,
        blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    scene.add(particles);

    // ---- 3D Football (Icosahedron) ----
    const fbGroup = new THREE.Group();
    const fbGeo = new THREE.IcosahedronGeometry(2.8, 1);
    fbGroup.add(new THREE.Mesh(fbGeo, new THREE.MeshPhongMaterial({
        color: 0x0a0a2e, emissive: 0x00d4ff, emissiveIntensity: 0.12,
        transparent: true, opacity: 0.35, shininess: 120,
    })));
    fbGroup.add(new THREE.Mesh(fbGeo.clone(), new THREE.MeshBasicMaterial({
        color: 0x00d4ff, wireframe: true, transparent: true, opacity: 0.45,
    })));

    // Rings
    const ringGeo = new THREE.TorusGeometry(4, 0.04, 16, 120);
    const ringMat = () => new THREE.MeshBasicMaterial({
        color: 0x00d4ff, transparent: true, opacity: 0.22,
    });
    const ring1 = new THREE.Mesh(ringGeo, ringMat());
    const ring2 = new THREE.Mesh(ringGeo.clone(), ringMat());
    ring2.rotation.set(Math.PI / 3, Math.PI / 5, 0);
    const ring3 = new THREE.Mesh(ringGeo.clone(), ringMat());
    ring3.rotation.set(-Math.PI / 4, 0, Math.PI / 6);
    ring3.material.color.setHex(0x7c3aed); ring3.material.opacity = 0.15;
    fbGroup.add(ring1, ring2, ring3);
    fbGroup.position.set(12, 1.5, -10);
    scene.add(fbGroup);

    // ---- Floating Shapes ----
    const shapes = [];
    const shapeDefs = [
        { geo: new THREE.OctahedronGeometry(0.9), col: 0x7c3aed, pos: [-14, 6, -12] },
        { geo: new THREE.TetrahedronGeometry(0.65), col: 0xec38bc, pos: [-9, -5, -8] },
        { geo: new THREE.IcosahedronGeometry(0.5), col: 0xfbbf24, pos: [5, -7, -14] },
        { geo: new THREE.OctahedronGeometry(0.45), col: 0x4ade80, pos: [-16, -1, -18] },
        { geo: new THREE.DodecahedronGeometry(0.75), col: 0x00d4ff, pos: [16, -4, -12] },
        { geo: new THREE.TetrahedronGeometry(0.5), col: 0xf87171, pos: [8, 8, -16] },
        { geo: new THREE.IcosahedronGeometry(0.6), col: 0x818cf8, pos: [-6, 9, -20] },
    ];
    shapeDefs.forEach(d => {
        const m = new THREE.Mesh(d.geo, new THREE.MeshBasicMaterial({
            color: d.col, wireframe: true, transparent: true, opacity: 0.35,
        }));
        m.position.set(...d.pos);
        m.userData = { baseY: d.pos[1], spd: 0.3 + Math.random() * 0.7, rot: 0.004 + Math.random() * 0.012 };
        shapes.push(m);
        scene.add(m);
    });

    // ---- Lights ----
    scene.add(new THREE.AmbientLight(0x404060, 0.4));
    const ltCyan = new THREE.PointLight(0x00d4ff, 1.6, 55); ltCyan.position.set(12, 5, -5);
    const ltPurple = new THREE.PointLight(0x7c3aed, 1.0, 45); ltPurple.position.set(-12, 4, -10);
    const ltPink = new THREE.PointLight(0xec38bc, 0.7, 35); ltPink.position.set(0, -6, -5);
    scene.add(ltCyan, ltPurple, ltPink);

    camera.position.set(0, 0, 22);

    // ---- Mouse Parallax ----
    let mx = 0, my = 0;
    document.addEventListener('mousemove', e => {
        mx = (e.clientX / window.innerWidth - 0.5) * 2;
        my = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    // ---- Animate ----
    let t = 0;
    function animate() {
        requestAnimationFrame(animate);
        t += 0.008;
        particles.rotation.y += 0.00018;
        particles.rotation.x += 0.00008;
        fbGroup.rotation.y += 0.0025;
        fbGroup.rotation.x += 0.0008;
        fbGroup.position.y = 1.5 + Math.sin(t * 0.5) * 0.6;
        ring1.rotation.z += 0.004;
        ring2.rotation.z -= 0.003;
        ring3.rotation.x += 0.002;
        shapes.forEach(s => {
            s.rotation.x += s.userData.rot;
            s.rotation.y += s.userData.rot * 0.7;
            s.position.y = s.userData.baseY + Math.sin(t * s.userData.spd) * 0.9;
        });
        camera.position.x += (mx * 2.5 - camera.position.x) * 0.015;
        camera.position.y += (-my * 1.5 - camera.position.y) * 0.015;
        camera.lookAt(0, 0, -5);
        ltCyan.intensity = 1.6 + Math.sin(t * 0.8) * 0.25;
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
})();


// =====================================================
// 2. APPLICATION STATE
// =====================================================
const APP = {
    taskId: null,
    pipelineDone: false,
    results: null,
    featuresFile: null,
    videoFile: null,
    qaMessages: [
        { role: 'ai', text: '👋 Welcome! I am your AI RAG Coordinator Agent. I can help explain the codebase design and architecture, or answer questions about any processed game timeline logs. Ask me anything!' }
    ],
    chatMessages: [],
    pollTimer: null,
};

// =====================================================
// 3. DOM REFERENCES
// =====================================================
const $ = id => document.getElementById(id);
const DOM = {
    tabsPre: $('tabs-pre'),
    tabsPost: $('tabs-post'),
    modelStatus: $('model-status'),
    threshold: $('threshold'),
    thresholdVal: $('threshold-val'),
    maxDuration: $('max-duration'),
    durationVal: $('duration-val'),
    enableAudio: $('enable-audio'),
    enableOcr: $('enable-ocr'),
    enableClip: $('enable-clip'),
    ocrPos: $('ocr-position'),
    hfToken: $('hf-token'),
    featDZ: $('features-dropzone'),
    vidDZ: $('video-dropzone'),
    featInput: $('features-input'),
    vidInput: $('video-input'),
    featName: $('features-filename'),
    vidName: $('video-filename'),
    runBtn: $('run-btn'),
    emptyState: $('empty-state'),
    loading: $('loading-overlay'),
    progFill: $('progress-fill'),
    progText: $('progress-text'),
    progPct: $('progress-percent'),
    metricsRow: $('metrics-row'),
    hlVideo: $('highlights-video'),
    dlBtn: $('download-btn'),
    summary: $('match-summary'),
    timelineCanvas: $('timeline-canvas'),
    eventsTable: $('events-table-container'),
    ocrTable: $('ocr-table-container'),
    ocrMilestones: $('ocr-milestones'),
    audioCanvas: $('audio-canvas'),
    audioTable: $('audio-table-container'),
    searchInput: $('search-input'),
    searchBtn: $('search-btn'),
    searchResults: $('search-results'),
    qaMessages: $('qa-messages'),
    qaInput: $('qa-input'),
    qaSend: $('qa-send'),
    chatMessages: $('chat-messages'),
    chatInput: $('chat-input'),
    chatSend: $('chat-send'),
};

// =====================================================
// 4. INITIALIZATION
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    initSliders();
    initDropZones();
    initTabs();
    initChat();
    renderQAMessages();
    checkLastJob();
});

async function checkLastJob() {
    const urlParams = new URLSearchParams(window.location.search);
    let taskId = urlParams.get('job_id');
    if (!taskId) {
        taskId = localStorage.getItem('last_task_id');
    }
    if (taskId) {
        try {
            const r = await fetch(API('/api/progress/' + taskId));
            if (r.ok) {
                const d = await r.json();
                if (d && !d.error) {
                    APP.taskId = taskId;
                    localStorage.setItem('last_task_id', taskId);

                    const queryJobId = urlParams.get('job_id');
                    if (queryJobId !== taskId) {
                        const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?job_id=' + taskId;
                        window.history.pushState({ path: newUrl }, '', newUrl);
                    }

                    if (d.done) {
                        DOM.loading.classList.remove('hidden');
                        DOM.progFill.style.width = '100%';
                        DOM.progText.textContent = 'Loading results...';
                        DOM.progPct.textContent = '100%';
                        await loadResults();
                    } else {
                        DOM.loading.classList.remove('hidden');
                        pollProgress();
                    }
                }
            } else {
                localStorage.removeItem('last_task_id');
            }
        } catch (e) {
            console.error('Failed to check last job status:', e);
        }
    }
}

// =====================================================
// 5. API HELPERS
// =====================================================
const API = path => path; // Same origin

async function checkStatus() {
    try {
        const r = await fetch(API('/api/status'));
        const d = await r.json();
        if (d.model && d.model.loaded) {
            DOM.modelStatus.innerHTML = '<span class="status-badge status-ready">✓ BiLSTM Model Loaded</span>';
        } else {
            DOM.modelStatus.innerHTML = '<span class="status-badge status-warn">⚠ Checkpoint Not Found</span>';
        }
    } catch {
        DOM.modelStatus.innerHTML = '<span class="status-badge status-warn">⚠ Server Unreachable</span>';
    }
}

// =====================================================
// 6. SLIDERS
// =====================================================
function initSliders() {
    DOM.threshold.addEventListener('input', () => DOM.thresholdVal.textContent = parseFloat(DOM.threshold.value).toFixed(2));
    DOM.maxDuration.addEventListener('input', () => DOM.durationVal.textContent = DOM.maxDuration.value);
}

// =====================================================
// 7. TAB NAVIGATION
// =====================================================
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const nav = btn.closest('.tabs-nav');
            nav.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const tab = btn.dataset.tab;
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            const panel = document.getElementById('panel-' + tab);
            if (panel) panel.classList.add('active');

            // Redraw canvas charts after the panel becomes visible
            if (APP.results) {
                requestAnimationFrame(() => {
                    if (tab === 'highlights') {
                        drawTimeline(APP.results.selected_events || [], APP.results.video_duration_min || 1);
                    } else if (tab === 'audio') {
                        const audio = APP.results.audio_timeline || [];
                        if (audio.length) drawAudioChart(audio, APP.results.video_duration_min || 1);
                    }
                });
            }
        });
    });
}

function switchToResults() {
    DOM.tabsPre.classList.add('hidden');
    DOM.tabsPost.classList.remove('hidden');
    // Activate first post-pipeline tab
    DOM.tabsPost.querySelector('.tab-btn').click();
    APP.pipelineDone = true;
}

// =====================================================
// 8. FILE UPLOAD (Drag & Drop)
// =====================================================
function initDropZones() {
    setupDZ(DOM.featDZ, DOM.featInput, DOM.featName, f => { APP.featuresFile = f; updateRunBtn(); });
    setupDZ(DOM.vidDZ, DOM.vidInput, DOM.vidName, f => { APP.videoFile = f; updateRunBtn(); });
    DOM.runBtn.addEventListener('click', startPipeline);
}

function setupDZ(dz, input, nameEl, onFile) {
    dz.addEventListener('click', () => input.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], nameEl, onFile);
    });
    input.addEventListener('change', () => { if (input.files.length) handleFile(input.files[0], nameEl, onFile); });
}

function handleFile(file, nameEl, onFile) {
    nameEl.textContent = '📎 ' + file.name + ' (' + (file.size / 1024 / 1024).toFixed(1) + ' MB)';
    nameEl.classList.remove('hidden');
    onFile(file);
}

function updateRunBtn() {
    DOM.runBtn.disabled = !(APP.featuresFile && APP.videoFile);
    DOM.emptyState.style.display = (APP.featuresFile && APP.videoFile) ? 'none' : 'block';
}

// =====================================================
// 9. PIPELINE EXECUTION
// =====================================================
async function startPipeline() {
    const form = new FormData();
    form.append('features', APP.featuresFile);
    form.append('video', APP.videoFile);
    form.append('threshold', DOM.threshold.value);
    form.append('max_duration', DOM.maxDuration.value);
    form.append('enable_audio', DOM.enableAudio.checked);
    form.append('enable_ocr', DOM.enableOcr.checked);
    form.append('enable_clip', DOM.enableClip.checked);
    form.append('ocr_crop_pos', DOM.ocrPos.value);
    form.append('hf_token', DOM.hfToken.value);

    DOM.loading.classList.remove('hidden');
    DOM.progFill.style.width = '0%';
    DOM.progText.textContent = 'Uploading files…';
    DOM.progPct.textContent = '0%';

    try {
        const r = await fetch(API('/api/run-pipeline'), { method: 'POST', body: form });
        const d = await r.json();
        if (d.error) { alert('Error: ' + d.error); DOM.loading.classList.add('hidden'); return; }
        APP.taskId = d.task_id;
        localStorage.setItem('last_task_id', d.task_id);
        const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?job_id=' + d.task_id;
        window.history.pushState({ path: newUrl }, '', newUrl);
        pollProgress();
    } catch (e) {
        alert('Failed to start pipeline: ' + e.message);
        DOM.loading.classList.add('hidden');
    }
}

function pollProgress() {
    APP.pollTimer = setInterval(async () => {
        try {
            const r = await fetch(API('/api/progress/' + APP.taskId));
            const d = await r.json();
            const pct = Math.round((d.progress || 0) * 100);
            DOM.progFill.style.width = pct + '%';
            DOM.progText.textContent = d.step || '';
            DOM.progPct.textContent = pct + '%';

            if (d.done) {
                clearInterval(APP.pollTimer);
                if (d.error) {
                    alert('Pipeline error: ' + d.error);
                    DOM.loading.classList.add('hidden');
                } else {
                    await loadResults();
                }
            }
        } catch { /* retry next tick */ }
    }, 800);
}

async function loadResults() {
    try {
        const r = await fetch(API('/api/results/' + APP.taskId));
        APP.results = await r.json();
        DOM.loading.classList.add('hidden');
        switchToResults();
        renderHighlights();
        renderOCR();
        renderAudio();
        renderChatWelcome();
        addTiltEffect();
    } catch (e) {
        alert('Failed to load results: ' + e.message);
        DOM.loading.classList.add('hidden');
    }
}

// =====================================================
// 10. RENDER: HIGHLIGHTS TAB
// =====================================================
function renderHighlights() {
    const R = APP.results;
    const sel = R.selected_events || [];
    const totalSec = sel.reduce((s, e) => s + (e.end - e.start) / 2, 0);
    const avgConf = sel.length ? sel.reduce((s, e) => s + e.confidence, 0) / sel.length : 0;

    // Metrics
    DOM.metricsRow.innerHTML = [
        metricCard('Highlights Spotted', sel.length + ' Clips'),
        metricCard('Highlight Duration', (totalSec / 60).toFixed(1) + ' Mins'),
        metricCard('Avg Confidence', (avgConf * 100).toFixed(1) + '%'),
        metricCard('Original Video', (R.video_duration_min || 0).toFixed(1) + ' Mins'),
    ].join('');

    // Video
    if (R.has_video && R.video_filename) {
        const url = API('/api/video/' + R.video_filename);
        DOM.hlVideo.src = url;
        DOM.dlBtn.href = url;
        DOM.dlBtn.style.display = 'block';
    } else {
        DOM.hlVideo.style.display = 'none';
        DOM.dlBtn.style.display = 'none';
    }

    // Summary
    DOM.summary.textContent = R.summary || 'No summary available.';

    // Timeline chart
    drawTimeline(sel, R.video_duration_min || 1);

    // Events table
    DOM.eventsTable.innerHTML = buildTable(
        ['Clip #', 'Event Type', 'Time (Min)', 'Duration', 'Confidence'],
        sel.map((e, i) => [
            i + 1,
            e.class || 'Visual Event',
            e.time_str || 'N/A',
            ((e.end - e.start) / 2).toFixed(0) + 's',
            (e.confidence * 100).toFixed(1) + '%',
        ])
    );

    // Render match frame gallery — 10-12 screenshots with timestamps only
    renderMatchTimeline(sel, R);
}

function renderMatchTimeline(events, results) {
    const container = $('match-timeline');
    if (!container) return;

    const tid = APP.taskId;
    const durationSec = (results.video_duration_min || 1) * 60;

    // Build timeline milestones list
    const milestones = [];

    // 1. Kick-off milestone
    milestones.push({
        time: 0,
        label: 'KICK-OFF',
        type: 'system'
    });

    // 2. Event milestones
    // Sort events by timestamp
    const sortedEvents = [...events].sort((a, b) => a.start - b.start);

    // Find first event of 2nd half if any (to insert HALF TIME chronologically)
    let halfTimeInserted = false;
    const halfTimeSec = durationSec / 2;

    sortedEvents.forEach(e => {
        const eventSec = e.time_seconds || (e.start / 2);

        // If we cross half time or event specifies half 2, insert HALF TIME milestone
        if (!halfTimeInserted && (eventSec >= halfTimeSec || e.half === 2)) {
            milestones.push({
                time: halfTimeSec,
                label: 'HALF TIME',
                type: 'system'
            });
            halfTimeInserted = true;
        }

        const min = Math.floor(eventSec / 60);
        const eventClass = e.class || 'Highlight';
        milestones.push({
            time: eventSec,
            label: `${min} MINUTE - ${eventClass.toUpperCase()}`,
            type: 'event'
        });
    });

    // If half time wasn't inserted, insert it now
    if (!halfTimeInserted) {
        milestones.push({
            time: halfTimeSec,
            label: 'HALF TIME',
            type: 'system'
        });
    }

    // 3. End of Match milestone
    milestones.push({
        time: durationSec,
        label: 'END OF MATCH',
        type: 'system'
    });

    // Deduplicate milestones that are too close (within 15 seconds) to keep the timeline clean
    const filteredMilestones = [];
    milestones.forEach(m => {
        const tooClose = filteredMilestones.some(existing =>
            Math.abs(existing.time - m.time) < 15 && existing.label !== 'HALF TIME' && m.label !== 'HALF TIME'
        );
        if (!tooClose) {
            filteredMilestones.push(m);
        }
    });

    // Sort final milestones chronologically
    filteredMilestones.sort((a, b) => a.time - b.time);

    const frameUrl = (sec) => API(`/api/frame/${tid}?t=${sec}&w=720`);
    const fmt = (sec) => Math.floor(sec / 60) + ':' + String(Math.floor(sec % 60)).padStart(2, '0');

    let html = '<div class="timeline-vertical-flow">';

    filteredMilestones.forEach((m) => {
        // We want 3 frames for each milestone:
        // - Main frame at T seconds
        // - Sub1 frame at T - 3 seconds
        // - Sub2 frame at T + 3 seconds
        // (Ensure they are within 0 and durationSec)
        let tMain = Math.round(m.time);
        let tSub1, tSub2;
        if (m.label === 'KICK-OFF') {
            tSub1 = Math.round(Math.min(durationSec, m.time + 2));
            tSub2 = Math.round(Math.min(durationSec, m.time + 4));
        } else if (m.label === 'END OF MATCH') {
            tMain = Math.round(Math.max(0, m.time - 2));
            tSub1 = Math.round(Math.max(0, m.time - 6));
            tSub2 = Math.round(Math.max(0, m.time - 4));
        } else {
            tSub1 = Math.round(Math.max(0, m.time - 3));
            tSub2 = Math.round(Math.min(durationSec, m.time + 3));
        }

        html += `
        <div class="timeline-node-container">
            <!-- Timeline Milestone Divider -->
            <div class="timeline-milestone">
                <div class="milestone-content">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="milestone-icon">
                        <circle cx="12" cy="13" r="8"></circle>
                        <path d="M12 9v4l2 2"></path>
                        <path d="M12 2v3"></path>
                        <path d="M9 2h6"></path>
                    </svg>
                    <span class="milestone-label">${m.label}</span>
                </div>
            </div>

            <!-- Milestone Photo Gallery (1 Big, 2 Small Layout) -->
            <div class="milestone-gallery">
                <div class="gallery-main">
                    <img src="${frameUrl(tMain)}" alt="Frame at ${fmt(tMain)}" loading="lazy"
                         onload="this.classList.add('loaded')">
                    <span class="gallery-ts">${fmt(tMain)}</span>
                </div>
                <div class="gallery-sub">
                    <div class="gallery-sub-item">
                        <img src="${frameUrl(tSub1)}" alt="Frame at ${fmt(tSub1)}" loading="lazy"
                             onload="this.classList.add('loaded')">
                        <span class="gallery-ts">${fmt(tSub1)}</span>
                    </div>
                    <div class="gallery-sub-item">
                        <img src="${frameUrl(tSub2)}" alt="Frame at ${fmt(tSub2)}" loading="lazy"
                             onload="this.classList.add('loaded')">
                        <span class="gallery-ts">${fmt(tSub2)}</span>
                    </div>
                </div>
            </div>
        </div>`;
    });

    html += '</div>';
    container.innerHTML = html;
}


function metricCard(label, value) {
    return `<div class="metric-card"><h4>${label}</h4><div class="value">${value}</div></div>`;
}

// =====================================================
// 11. RENDER: OCR TAB
// =====================================================
function renderOCR() {
    const ocr = APP.results.ocr_timeline || [];
    if (!ocr.length) {
        DOM.ocrTable.innerHTML = '<p style="color:var(--text-muted)">Scoreboard OCR was disabled or no scores were detected.</p>';
        DOM.ocrMilestones.innerHTML = '';
        return;
    }
    DOM.ocrTable.innerHTML = buildTable(
        ['Match Timestamp', 'Scoreboard Time', 'Current Score'],
        ocr.map(e => [
            Math.floor(e.time_seconds / 60) + ':' + String(Math.floor(e.time_seconds % 60)).padStart(2, '0'),
            e.game_time_str || 'N/A',
            e.score_str || 'N/A',
        ])
    );
    // Milestones
    let last = null;
    const milestones = ocr.filter(e => { if (e.score_str && e.score_str !== last) { last = e.score_str; return true; } return false; });
    DOM.ocrMilestones.innerHTML = milestones.length
        ? milestones.map((m, i) => `<div class="milestone-card">🏆 <strong>Milestone ${i + 1}</strong>: ${m.score_str} at match minute ${m.game_time_str || 'N/A'}</div>`).join('')
        : '<p style="color:var(--text-muted)">No score change milestones recorded.</p>';
}

// =====================================================
// 12. RENDER: AUDIO TAB
// =====================================================
function renderAudio() {
    const audio = APP.results.audio_timeline || [];
    if (!audio.length) {
        DOM.audioTable.innerHTML = '<p style="color:var(--text-muted)">Audio classifier was disabled or no events detected.</p>';
        return;
    }
    drawAudioChart(audio, APP.results.video_duration_min || 1);
    DOM.audioTable.innerHTML = buildTable(
        ['Timestamp', 'Event Class', 'Confidence'],
        audio.map(e => [
            Math.floor(e.time_seconds / 60) + ':' + String(Math.floor(e.time_seconds % 60)).padStart(2, '0'),
            e.class,
            (e.confidence * 100).toFixed(1) + '%',
        ])
    );
}

// =====================================================
// 13. RENDER: CHAT WELCOME
// =====================================================
function renderChatWelcome() {
    APP.chatMessages = [
        { role: 'ai', text: '👋 Welcome! I can answer questions about the processed match (goals, cards, timeline) or explain the codebase architecture. Ask me anything!' }
    ];
    renderChatMessages();
}

// =====================================================
// 14. CANVAS CHARTS
// =====================================================
function drawTimeline(events, durationMin) {
    const canvas = DOM.timelineCanvas;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = 100 * dpr;
    canvas.style.height = '100px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const w = canvas.offsetWidth, h = 100;

    ctx.fillStyle = 'rgba(14,17,32,0.65)';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(74,74,106,0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    // Bars
    const colors = { Goal: 'rgba(0,212,255,', Cards: 'rgba(251,191,36,', Substitution: 'rgba(192,3,189,' };
    events.forEach(e => {
        const x1 = (e.start / 2 / 60 / durationMin) * w;
        const x2 = (e.end / 2 / 60 / durationMin) * w;
        const base = colors[e.class] || 'rgba(99,102,241,';
        ctx.fillStyle = base + (0.4 + 0.6 * e.confidence) + ')';
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 0.5;
        const barH = 30;
        const y = (h - barH) / 2;
        ctx.beginPath();
        ctx.roundRect(x1, y, Math.max(x2 - x1, 3), barH, 4);
        ctx.fill(); ctx.stroke();
    });

    // X-axis labels
    ctx.fillStyle = '#a0a0cc'; ctx.font = '11px Outfit';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
        const min = ((i / 5) * durationMin).toFixed(0);
        ctx.fillText(min + "'", (i / 5) * w, h - 6);
    }
}

function drawAudioChart(events, durationMin) {
    const canvas = DOM.audioCanvas;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = 300 * dpr;
    canvas.style.height = '300px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const w = canvas.offsetWidth, h = 300;
    const pad = { top: 20, bottom: 40, left: 50, right: 20 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    ctx.fillStyle = 'rgba(14,17,32,0.65)';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(74,74,106,0.12)';
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (i / 5) * ch;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = '#a0a0cc'; ctx.font = '11px Outfit';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
        const min = ((i / 5) * durationMin).toFixed(0);
        ctx.fillText(min + "'", pad.left + (i / 5) * cw, h - pad.bottom + 18);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const v = (1 - i / 5).toFixed(1);
        ctx.fillText(v, pad.left - 8, pad.top + (i / 5) * ch + 4);
    }

    // Dots
    events.forEach(e => {
        const x = pad.left + (e.time_seconds / 60 / durationMin) * cw;
        const y = pad.top + (1 - e.confidence) * ch;
        const isWhistle = e.class === 'Referee Whistle';

        ctx.save();
        // Glow
        ctx.shadowColor = isWhistle ? '#fbbf24' : '#00d4ff';
        ctx.shadowBlur = 12;
        ctx.fillStyle = isWhistle ? '#fbbf24' : '#00d4ff';

        if (isWhistle) {
            // Triangle
            ctx.beginPath();
            ctx.moveTo(x, y - 8); ctx.lineTo(x - 7, y + 6); ctx.lineTo(x + 7, y + 6);
            ctx.closePath(); ctx.fill();
            ctx.strokeStyle = '#000'; ctx.lineWidth = 0.5; ctx.stroke();
        } else {
            // Circle
            ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#fff'; ctx.lineWidth = 0.5; ctx.stroke();
        }
        ctx.restore();
    });

    // Legend
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#00d4ff'; ctx.beginPath(); ctx.arc(w - 180, h - 14, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#a0a0cc'; ctx.font = '11px Outfit'; ctx.textAlign = 'left';
    ctx.fillText('Crowd Cheer', w - 170, h - 10);
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath(); ctx.moveTo(w - 80, h - 19); ctx.lineTo(w - 87, h - 7); ctx.lineTo(w - 73, h - 7); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#a0a0cc';
    ctx.fillText('Whistle', w - 68, h - 10);
}

// =====================================================
// 15. CHAT SYSTEM
// =====================================================
function initChat() {
    // Q&A (pre-pipeline)
    DOM.qaSend.addEventListener('click', () => sendQA());
    DOM.qaInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendQA(); });

    // Full chatbot (post-pipeline)
    DOM.chatSend.addEventListener('click', () => sendChat());
    DOM.chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });

    // CLIP Search
    DOM.searchBtn.addEventListener('click', () => doSearch());
    DOM.searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });
}

async function sendQA() {
    const msg = DOM.qaInput.value.trim();
    if (!msg) return;
    APP.qaMessages.push({ role: 'user', text: msg });
    DOM.qaInput.value = '';
    renderQAMessages();
    try {
        const r = await fetch(API('/api/chat'), {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, task_id: APP.taskId || '' }),
        });
        const d = await r.json();
        APP.qaMessages.push({ role: 'ai', text: d.response });
    } catch {
        APP.qaMessages.push({ role: 'ai', text: '⚠️ Failed to reach the server.' });
    }
    renderQAMessages();
}

async function sendChat() {
    const msg = DOM.chatInput.value.trim();
    if (!msg) return;
    APP.chatMessages.push({ role: 'user', text: msg });
    DOM.chatInput.value = '';
    renderChatMessages();
    try {
        const r = await fetch(API('/api/chat'), {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, task_id: APP.taskId || '' }),
        });
        const d = await r.json();
        APP.chatMessages.push({ role: 'ai', text: d.response });
    } catch {
        APP.chatMessages.push({ role: 'ai', text: '⚠️ Failed to reach the server.' });
    }
    renderChatMessages();
}

function renderQAMessages() {
    DOM.qaMessages.innerHTML = APP.qaMessages.map(m => chatBubble(m)).join('');
    DOM.qaMessages.scrollTop = DOM.qaMessages.scrollHeight;
}

function renderChatMessages() {
    DOM.chatMessages.innerHTML = APP.chatMessages.map(m => chatBubble(m)).join('');
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
}

function chatBubble(m) {
    const cls = m.role === 'user' ? 'bubble-user' : 'bubble-ai';
    const icon = m.role === 'user' ? '👤' : '🧠';
    const label = m.role === 'user' ? 'You' : 'AI Assistant';
    return `<div class="chat-bubble ${cls}"><div class="bubble-role">${icon} ${label}</div><div>${escapeHTML(m.text)}</div></div>`;
}

function escapeHTML(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

// =====================================================
// 16. CLIP SEARCH
// =====================================================
async function doSearch() {
    const query = DOM.searchInput.value.trim();
    if (!query || !APP.taskId) return;
    DOM.searchResults.innerHTML = '<p style="color:var(--text-muted);text-align:center">Searching…</p>';
    try {
        const r = await fetch(API('/api/search'), {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, task_id: APP.taskId, top_k: 4 }),
        });
        const d = await r.json();
        if (d.error) { DOM.searchResults.innerHTML = `<p style="color:var(--accent-gold)">${d.error}</p>`; return; }
        if (!d.matches || !d.matches.length) {
            DOM.searchResults.innerHTML = '<p style="color:var(--accent-gold)">No matches found above similarity threshold.</p>';
            return;
        }
        DOM.searchResults.innerHTML = d.matches.map(m => {
            const t = m.timestamp || 0;
            const min = Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0');
            return `<div class="search-card">
                <h4>Match Time: ${min}</h4>
                <p class="score">Similarity Score: ${(m.score || 0).toFixed(2)}</p>
                <p class="context">Context: ${m.event_class || 'Visual Frame'}</p>
            </div>`;
        }).join('');
    } catch {
        DOM.searchResults.innerHTML = '<p style="color:var(--accent-red)">Search failed.</p>';
    }
}

// =====================================================
// 17. TABLE BUILDER
// =====================================================
function buildTable(headers, rows) {
    if (!rows.length) return '<p style="color:var(--text-muted)">No data available.</p>';
    return `<table class="data-table">
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>
    </table>`;
}

// =====================================================
// 18. 3D CARD TILT EFFECT
// =====================================================
function addTiltEffect() {
    document.querySelectorAll('.metric-card, .search-card').forEach(el => {
        el.addEventListener('mousemove', e => {
            const rect = el.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            el.style.transform = `perspective(800px) rotateY(${x * 12}deg) rotateX(${-y * 12}deg) translateZ(8px) translateY(-4px)`;
        });
        el.addEventListener('mouseleave', () => {
            el.style.transform = 'perspective(800px) rotateY(0) rotateX(0) translateZ(0) translateY(0)';
            el.style.transition = 'transform 0.5s ease';
        });
        el.addEventListener('mouseenter', () => { el.style.transition = 'transform 0.1s ease'; });
    });
}
