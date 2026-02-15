/**
 * Volleyball Spike Detector Platform - Frontend
 *
 * Single-page vanilla JS application with 3 views:
 *   1. Video Manager - upload & list videos
 *   2. Segment Review - label segments
 *   3. Training Dashboard - train & track models
 */

const API = '/api';

// ─── State ────────────────────────────────────────────────────────

const PHASE_NAMES = ['approach', 'jump', 'swing', 'land'];
const PHASE_COLORS = { approach: '#3b82f6', jump: '#22c55e', swing: '#f59e0b', land: '#ef4444' };

const state = {
    currentView: 'videos',
    // Review state
    reviewVideoId: null,
    reviewVideoFps: 30,
    reviewVideoWidth: 0,
    reviewVideoHeight: 0,
    segments: [],
    selectedSegmentIdx: -1,
    segmentFilter: 'all',
    // Bbox overlay
    trackBboxes: [],   // [{frame, bbox: [x1,y1,x2,y2]}, ...]
    _bboxAnimFrame: null,
    // Polling intervals
    _pollInterval: null,
    // Phase annotation state
    reviewMode: 'spike',   // 'spike' | 'phase' | 'role'
    phaseFrames: [],       // extended frames [{frame_number, bbox}]
    phaseBoundaries: {},   // { approach: frame, jump: frame, swing: frame, land: frame }
    phaseSegmentId: null,  // segment being phase-annotated
    // Track-based grouping for phase mode
    phaseTracks: [],       // [{track_id, segments: [...], annotatedCount, predictedCount, totalCount, minTime, maxTime}]
    selectedTrackIdx: -1,
    phaseTrackSegments: [], // all spike segment IDs for selected track
    phasePredicted: false,  // true if showing model predictions (not human labels)
    // Role ID mode
    roleTracks: [],            // track objects from API (all)
    filteredRoleTracks: [],    // after role filter applied
    selectedRoleTrackIdx: -1,
    roleFilter: 'all',         // 'all' | 'player' | 'non_player' | 'unknown'
};

// ─── Navigation ───────────────────────────────────────────────────

document.querySelectorAll('nav button').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
});

function switchView(view) {
    state.currentView = view;
    document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
    document.querySelector(`nav button[data-view="${view}"]`).classList.add('active');
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');

    if (view === 'videos') loadVideos();
    if (view === 'review') loadReviewVideoList();
    if (view === 'training') { loadTrainingRuns(); loadLabelStats(); loadGroupMetricsRunSelector(); }
}

// ─── API Helpers ──────────────────────────────────────────────────

async function api(path, options = {}) {
    const resp = await fetch(`${API}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || resp.statusText);
    }
    if (resp.status === 204) return null;
    return resp.json();
}

// ─── View 1: Video Manager ───────────────────────────────────────

const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadVideo(fileInput.files[0]);
});

function uploadVideo(file) {
    const progressEl = document.getElementById('upload-progress');
    const barEl = document.getElementById('upload-bar');
    const pctEl = document.getElementById('upload-pct');
    const nameEl = document.getElementById('upload-filename');

    nameEl.textContent = file.name;
    barEl.style.width = '0%';
    pctEl.textContent = '0%';
    progressEl.style.display = 'block';
    uploadArea.style.display = 'none';

    const form = new FormData();
    form.append('file', file);

    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', e => {
        if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            barEl.style.width = pct + '%';
            pctEl.textContent = pct + '%';
        }
    });

    xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            // Show 100% briefly before hiding
            barEl.style.width = '100%';
            pctEl.textContent = '100%';
            setTimeout(() => {
                progressEl.style.display = 'none';
                uploadArea.style.display = '';
                loadVideos();
            }, 500);
        } else {
            progressEl.style.display = 'none';
            uploadArea.style.display = '';
            alert('Upload failed: ' + xhr.statusText);
        }
    });

    xhr.addEventListener('error', () => {
        progressEl.style.display = 'none';
        uploadArea.style.display = '';
        alert('Upload failed: network error');
    });

    xhr.open('POST', `${API}/videos`);
    xhr.send(form);
}

async function loadVideos() {
    const videos = await api('/videos');
    const tbody = document.getElementById('video-table-body');
    tbody.innerHTML = '';

    const VIDEO_GROUPS = ['hitting_lines', 'game_play', 'scrimmage'];

    for (const v of videos) {
        const tr = document.createElement('tr');
        const duration = v.duration_seconds ? `${Math.round(v.duration_seconds)}s` : '-';

        // Build group dropdown
        let groupOptions = '<option value="">—</option>';
        for (const g of VIDEO_GROUPS) {
            groupOptions += `<option value="${g}" ${v.video_group === g ? 'selected' : ''}>${g.replace(/_/g, ' ')}</option>`;
        }
        // If current group is custom (not in predefined list), add it
        if (v.video_group && !VIDEO_GROUPS.includes(v.video_group)) {
            groupOptions += `<option value="${esc(v.video_group)}" selected>${esc(v.video_group)}</option>`;
        }
        groupOptions += '<option value="__custom__">Custom...</option>';

        tr.innerHTML = `
            <td>${esc(v.filename)}</td>
            <td>${duration}</td>
            <td>
                <span class="badge badge-${v.status}">${v.status}</span>
                ${v.status === 'processing' ? `
                    <div id="progress-${v.id}" style="margin-top:4px;">
                        <div class="processing-bar"><div class="fill"></div></div>
                        <div class="progress-text" style="font-size:11px; color:var(--text-muted); margin-top:2px;"></div>
                    </div>
                ` : ''}
            </td>
            <td>
                <select class="video-group-select" data-video-id="${v.id}" style="padding:4px 6px; background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:var(--radius); font-size:12px; max-width:120px;">
                    ${groupOptions}
                </select>
            </td>
            <td>${v.track_count}</td>
            <td>${v.segment_count}</td>
            <td>${v.labeled_count}/${v.segment_count}</td>
            <td>
                ${v.status === 'processed' || v.status === 'predicted' || v.status === 'error' ? `<button class="btn btn-sm btn-primary" onclick="reprocessVideo('${v.id}')">Reprocess</button>` : ''}
                <button class="btn btn-sm btn-outline" onclick="openReview('${v.id}')">Review</button>
                <button class="btn btn-sm btn-danger" onclick="deleteVideo('${v.id}')">Delete</button>
            </td>
        `;
        tbody.appendChild(tr);

        // Auto-poll if processing
        if (v.status === 'processing' || v.status === 'uploaded') {
            startPolling(v.id);
        }
    }

    // Wire up group dropdowns
    tbody.querySelectorAll('.video-group-select').forEach(sel => {
        sel.addEventListener('change', async () => {
            const videoId = sel.dataset.videoId;
            let group = sel.value;
            if (group === '__custom__') {
                group = prompt('Enter custom group name:');
                if (!group) { sel.value = ''; return; }
            }
            try {
                await api(`/videos/${videoId}/group`, {
                    method: 'PATCH',
                    body: JSON.stringify({ video_group: group || null }),
                });
            } catch (e) {
                alert('Failed to set group: ' + e.message);
                loadVideos();
            }
        });
    });
}

async function reprocessVideo(id) {
    if (!confirm('This will delete existing tracks, segments, and labels for this video, then re-run detection and tracking. Continue?')) return;
    try {
        await api(`/videos/${id}/reprocess`, { method: 'POST' });
        loadVideos();
    } catch (e) {
        alert('Reprocess failed: ' + e.message);
    }
}

async function deleteVideo(id) {
    if (!confirm('Delete this video and all data?')) return;
    await api(`/videos/${id}`, { method: 'DELETE' });
    loadVideos();
}

function startPolling(videoId) {
    if (state._pollInterval) clearInterval(state._pollInterval);
    state._pollInterval = setInterval(async () => {
        try {
            const status = await api(`/videos/${videoId}/status`);
            // Update progress bar in the table row
            const progressEl = document.getElementById(`progress-${videoId}`);
            if (progressEl && status.progress_pct != null) {
                progressEl.querySelector('.fill').style.width = status.progress_pct + '%';
                progressEl.querySelector('.progress-text').textContent =
                    status.message || `${Math.round(status.progress_pct)}%`;
            }
            if (status.status !== 'processing' && status.status !== 'uploaded') {
                clearInterval(state._pollInterval);
                state._pollInterval = null;
                loadVideos();
            }
        } catch { /* ignore */ }
    }, 2000);
}

// ─── View 2: Segment Review ──────────────────────────────────────

// Review mode toggle
document.getElementById('review-mode-toggle').addEventListener('click', e => {
    if (e.target.tagName !== 'BUTTON') return;
    const mode = e.target.dataset.reviewMode;
    if (!mode || mode === state.reviewMode) return;

    state.reviewMode = mode;
    document.querySelectorAll('#review-mode-toggle button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');

    // Toggle UI panels
    document.getElementById('spike-mode-controls').style.display = mode === 'spike' ? '' : 'none';
    document.getElementById('phase-mode-controls').style.display = mode === 'phase' ? '' : 'none';
    document.getElementById('role-mode-controls').style.display = mode === 'role' ? '' : 'none';
    document.getElementById('phase-annotation-panel').style.display = 'none';

    // Reset selection and reload segments for the new mode
    state.selectedSegmentIdx = -1;
    state.selectedTrackIdx = -1;
    state.selectedRoleTrackIdx = -1;
    state.phaseSegmentId = null;
    state.phaseBoundaries = {};
    state.phaseTrackSegments = [];
    state.trackBboxes = [];
    clearBbox();

    if (mode === 'phase') {
        // In phase mode, force filter to spike-only
        state.segmentFilter = 'spike';
    } else if (mode === 'spike') {
        state.segmentFilter = 'all';
        // Re-activate the "All" filter button
        document.querySelectorAll('#segment-filters button').forEach(b => b.classList.remove('active'));
        document.querySelector('#segment-filters button[data-filter="all"]').classList.add('active');
    }

    if (mode === 'role') {
        loadRoleTracks();
    } else {
        loadSegments();
    }
});

const reviewVideoSelect = document.getElementById('review-video-select');
reviewVideoSelect.addEventListener('change', () => {
    if (reviewVideoSelect.value) openReview(reviewVideoSelect.value);
});

async function loadReviewVideoList() {
    const videos = await api('/videos');
    reviewVideoSelect.innerHTML = '<option value="">Select a video to review...</option>';
    for (const v of videos) {
        if (v.segment_count > 0) {
            const opt = document.createElement('option');
            opt.value = v.id;
            opt.textContent = `${v.filename} (${v.segment_count} segments, ${v.labeled_count} labeled)`;
            if (v.id === state.reviewVideoId) opt.selected = true;
            reviewVideoSelect.appendChild(opt);
        }
    }
}

async function openReview(videoId) {
    switchView('review');
    state.reviewVideoId = videoId;

    const video = await api(`/videos/${videoId}`);
    state.reviewVideoFps = video.fps || 30;
    state.reviewVideoWidth = video.width;
    state.reviewVideoHeight = video.height;

    // Set video source
    const videoEl = document.getElementById('review-video');
    videoEl.src = `${API}/videos/${videoId}/clip`;

    // Update frame counter + bbox overlay on timeupdate
    videoEl.ontimeupdate = () => {
        const frame = Math.round(videoEl.currentTime * state.reviewVideoFps);
        document.getElementById('frame-counter').textContent = `Frame: ${frame}`;
        document.getElementById('time-display').textContent = `${videoEl.currentTime.toFixed(2)}s`;
        drawBbox(frame);
    };

    // Update select
    reviewVideoSelect.value = videoId;

    if (state.reviewMode === 'role') {
        loadRoleTracks();
    } else {
        loadSegments();
    }
}

async function loadSegments() {
    if (!state.reviewVideoId) return;

    let filter = '';
    if (state.reviewMode === 'phase') {
        // Phase mode: only show spike segments
        filter = '&label=1';
    } else {
        if (state.segmentFilter === 'spike') filter = '&label=1';
        else if (state.segmentFilter === 'non-spike') filter = '&label=0';
        else if (state.segmentFilter === 'unlabeled') filter = '&unlabeled_only=true';
    }

    // Always filter to player tracks (non-player excluded) in spike/phase modes
    filter += '&role=player';

    const data = await api(`/videos/${state.reviewVideoId}/segments?per_page=200${filter}`);
    state.segments = data.segments;

    // In phase mode, group by track and check annotation status
    if (state.reviewMode === 'phase') {
        // Check phase annotation status for each segment (human vs predicted)
        for (const seg of state.segments) {
            try {
                const phases = await api(`/segments/${seg.id}/phases`);
                seg._hasHumanPhases = phases.some(p => p.human_label != null);
                seg._hasPredictedPhases = phases.some(p => p.human_label == null && p.model_run_id != null);
                seg._hasPhases = phases.length > 0;
            } catch {
                seg._hasHumanPhases = false;
                seg._hasPredictedPhases = false;
                seg._hasPhases = false;
            }
        }

        // Group segments by track_id
        const trackMap = {};
        for (const seg of state.segments) {
            if (!trackMap[seg.track_id]) {
                trackMap[seg.track_id] = [];
            }
            trackMap[seg.track_id].push(seg);
        }

        state.phaseTracks = Object.entries(trackMap).map(([trackId, segs]) => {
            const annotatedCount = segs.filter(s => s._hasHumanPhases).length;
            const predictedCount = segs.filter(s => s._hasPredictedPhases && !s._hasHumanPhases).length;
            const minTime = Math.min(...segs.map(s => s.start_time || 0));
            const maxTime = Math.max(...segs.map(s => s.end_time || 0));
            return {
                track_id: parseInt(trackId),
                segments: segs,
                annotatedCount,
                predictedCount,
                totalCount: segs.length,
                minTime,
                maxTime,
            };
        });
        // Sort by time
        state.phaseTracks.sort((a, b) => a.minTime - b.minTime);
    }

    renderSegments();
    renderStats();
}

function renderSegments() {
    const list = document.getElementById('segment-list');
    list.innerHTML = '';

    if (state.reviewMode === 'phase') {
        // Phase mode: render track cards
        state.phaseTracks.forEach((track, idx) => {
            const card = document.createElement('div');
            card.className = 'segment-card' + (idx === state.selectedTrackIdx ? ' selected' : '');
            card.dataset.idx = idx;

            const startTime = track.minTime.toFixed(2);
            const endTime = track.maxTime.toFixed(2);

            let statusBadge;
            if (track.annotatedCount === track.totalCount) {
                statusBadge = '<span class="badge" style="background:var(--success); color:white;">annotated</span>';
            } else if (track.annotatedCount > 0) {
                statusBadge = `<span class="badge" style="background:var(--warning); color:black;">${track.annotatedCount}/${track.totalCount} annotated</span>`;
            } else if (track.predictedCount > 0) {
                statusBadge = '<span class="badge" style="background:var(--primary); color:white;">predicted</span>';
            } else {
                statusBadge = '<span class="badge" style="background:var(--border); color:var(--text-muted);">unannotated</span>';
            }

            card.innerHTML = `
                <div class="meta">
                    <span>Track ${track.track_id}</span>
                    ${statusBadge}
                </div>
                <div class="seg-fields">
                    <span class="seg-field"><span class="seg-label">${track.totalCount} spike segment${track.totalCount !== 1 ? 's' : ''}</span></span>
                    <span class="seg-field"><span class="seg-label">${startTime}s - ${endTime}s</span></span>
                </div>
            `;

            card.addEventListener('click', () => selectTrack(idx));
            list.appendChild(card);
        });
        return;
    }

    state.segments.forEach((seg, idx) => {
        const card = document.createElement('div');
        card.className = 'segment-card' + (idx === state.selectedSegmentIdx ? ' selected' : '');
        card.dataset.idx = idx;

        const startTime = seg.start_time != null ? seg.start_time.toFixed(2) : '?';
        const endTime = seg.end_time != null ? seg.end_time.toFixed(2) : '?';

        {
            // Spike mode: existing behavior
            let predHtml = '';
            if (seg.prediction != null) {
                const predName = seg.prediction === 1 ? 'spike' : 'non-spike';
                const predColor = seg.prediction === 1 ? 'var(--spike)' : 'var(--non-spike)';
                const conf = seg.confidence != null ? ` (${(seg.confidence * 100).toFixed(0)}%)` : '';
                predHtml = `<span class="seg-field"><span class="seg-label">Predicted</span> <span style="color:${predColor}; font-weight:600;">${predName}${conf}</span></span>`;
            }

            let labelHtml = '';
            if (seg.human_label != null) {
                const labelName = seg.human_label === 1 ? 'spike' : 'non-spike';
                const labelColor = seg.human_label === 1 ? 'var(--spike)' : 'var(--non-spike)';
                labelHtml = `<span class="seg-field"><span class="seg-label">Labeled</span> <span style="color:${labelColor}; font-weight:600;">${labelName}</span></span>`;
            } else {
                labelHtml = `<span class="seg-field"><span class="seg-label">Labeled</span> <span style="color:var(--text-muted);">-</span></span>`;
            }

            const mismatch = seg.prediction != null && seg.human_label != null && seg.prediction !== seg.human_label;

            card.innerHTML = `
                <div class="meta">
                    <span>Track ${seg.track_id} | ${startTime}s - ${endTime}s</span>
                    ${mismatch ? '<span class="badge" style="background:var(--warning); color:black;">mismatch</span>' : ''}
                </div>
                <div class="seg-fields">
                    ${predHtml}
                    ${labelHtml}
                </div>
                <div class="actions">
                    <button class="btn btn-sm btn-spike" onclick="labelSegment(${seg.id}, 1, event)">Spike</button>
                    <button class="btn btn-sm btn-non-spike" onclick="labelSegment(${seg.id}, 0, event)">Non-spike</button>
                </div>
            `;
        }

        card.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') return;
            selectSegment(idx);
        });

        list.appendChild(card);
    });
}

async function selectSegment(idx) {
    state.selectedSegmentIdx = idx;
    state.trackBboxes = [];
    clearBbox();
    renderSegments();

    const seg = state.segments[idx];
    if (!seg) return;

    if (seg.start_time != null) {
        const videoEl = document.getElementById('review-video');
        videoEl.currentTime = seg.start_time;
    }

    // Fetch bounding boxes for this segment's track
    try {
        state.trackBboxes = await api(
            `/videos/${state.reviewVideoId}/tracks/${seg.track_id}/bboxes?start_frame=${seg.start_frame}&end_frame=${seg.end_frame}`
        );
        drawBbox(Math.round((seg.start_time || 0) * state.reviewVideoFps));
    } catch { /* ignore */ }

    // In phase mode, track selection is handled by selectTrack() instead
    document.getElementById('phase-annotation-panel').style.display = 'none';
}

async function labelSegment(segId, label, event) {
    if (event) event.stopPropagation();
    const currentIdx = state.selectedSegmentIdx;
    await api(`/segments/${segId}`, {
        method: 'PATCH',
        body: JSON.stringify({ human_label: label }),
    });
    await loadSegments();
    // If a filter (e.g. "unlabeled") removed the labeled segment from the list,
    // all indices shifted down — the next segment is now at currentIdx.
    // If the segment is still in the list, the next one is at currentIdx + 1.
    const stillInList = state.segments.some(s => s.id === segId);
    const nextIdx = stillInList ? currentIdx + 1 : currentIdx;
    if (nextIdx < state.segments.length) {
        selectSegment(nextIdx);
        scrollSegmentIntoView();
    }
}

async function renderStats() {
    if (!state.reviewVideoId) return;
    // Fetch full stats from all segments of this video
    const allData = await api(`/videos/${state.reviewVideoId}/segments?per_page=1`);
    const total = allData.total;

    const labeledData = await api(`/videos/${state.reviewVideoId}/segments?per_page=1&unlabeled_only=false`);

    const bar = document.getElementById('stats-bar');
    const labeled = state.segments.filter(s => s.human_label != null).length;
    const spikes = state.segments.filter(s => s.human_label === 1).length;
    const nonSpikes = state.segments.filter(s => s.human_label === 0).length;

    bar.innerHTML = `
        <span>Total: <span class="stat-value">${total}</span></span>
        <span>Showing: <span class="stat-value">${state.segments.length}</span></span>
        <span>Labeled: <span class="stat-value">${labeled}</span></span>
        <span>Spike: <span class="stat-value" style="color:var(--spike);">${spikes}</span></span>
        <span>Non-spike: <span class="stat-value">${nonSpikes}</span></span>
    `;
}

// ─── Role ID Mode ─────────────────────────────────────────────────

async function loadRoleTracks() {
    if (!state.reviewVideoId) return;
    try {
        state.roleTracks = await api(`/videos/${state.reviewVideoId}/tracks`);
    } catch {
        state.roleTracks = [];
    }
    applyRoleFilter();
    renderRoleTracks();
    renderRoleStats();
}

function applyRoleFilter() {
    if (state.roleFilter === 'all') {
        state.filteredRoleTracks = state.roleTracks;
    } else {
        state.filteredRoleTracks = state.roleTracks.filter(t => (t.role || 'unknown') === state.roleFilter);
    }
    state.selectedRoleTrackIdx = -1;
}

function renderRoleTracks() {
    const list = document.getElementById('segment-list');
    list.innerHTML = '';
    const bar = document.getElementById('stats-bar');
    bar.innerHTML = '';

    state.filteredRoleTracks.forEach((track, idx) => {
        const card = document.createElement('div');
        card.className = 'segment-card' + (idx === state.selectedRoleTrackIdx ? ' selected' : '');
        card.dataset.idx = idx;

        const startTime = (track.start_frame / state.reviewVideoFps).toFixed(2);
        const endTime = (track.end_frame / state.reviewVideoFps).toFixed(2);

        const role = track.role || 'unknown';
        let roleBadgeStyle;
        if (role === 'player') roleBadgeStyle = 'background:var(--success); color:white;';
        else if (role === 'non_player') roleBadgeStyle = 'background:var(--danger, #ef4444); color:white;';
        else roleBadgeStyle = 'background:var(--border); color:var(--text-muted);';

        const sourceLabel = track.role_source === 'human' ? 'human' : track.role_source === 'model' ? 'model' : track.role_source === 'heuristic' ? 'auto' : '';
        const sourceBadge = sourceLabel ? `<span class="badge" style="background:var(--surface); color:var(--text-muted); font-size:10px;">${sourceLabel}</span>` : '';

        const bboxArea = track.median_bbox_area != null ? `bbox: ${(track.median_bbox_area * 100).toFixed(1)}%` : '';
        const poseConf = track.median_pose_confidence != null ? `pose: ${(track.median_pose_confidence * 100).toFixed(0)}%` : '';
        const moveVar = track.movement_variance != null ? `move: ${track.movement_variance.toFixed(4)}` : '';
        const vertRange = track.vertical_range != null ? `vert: ${(track.vertical_range * 100).toFixed(1)}%` : '';
        const statsStr = [bboxArea, poseConf, moveVar, vertRange].filter(Boolean).join(' | ');

        card.innerHTML = `
            <div class="meta">
                <span>Track ${track.track_id} &middot; ${track.frame_count} frames</span>
                <span style="display:flex; gap:4px; align-items:center;">
                    <span class="badge" style="${roleBadgeStyle}">${role.replace('_', '-')}</span>
                    ${sourceBadge}
                </span>
            </div>
            <div class="seg-fields">
                <span class="seg-field"><span class="seg-label">${startTime}s - ${endTime}s</span></span>
                ${statsStr ? `<span class="seg-field"><span class="seg-label" style="font-size:11px; color:var(--text-muted);">${statsStr}</span></span>` : ''}
                <span class="seg-field"><span class="seg-label">${track.segment_count} segment${track.segment_count !== 1 ? 's' : ''}</span></span>
            </div>
            <div class="actions">
                <button class="btn btn-sm btn-spike" onclick="setTrackRole(${track.id}, 'player', event)">Player</button>
                <button class="btn btn-sm btn-non-spike" onclick="setTrackRole(${track.id}, 'non_player', event)">Non-player</button>
            </div>
        `;

        card.addEventListener('click', (e) => {
            if (e.target.tagName === 'BUTTON') return;
            selectRoleTrack(idx);
        });

        list.appendChild(card);
    });
}

function renderRoleStats() {
    const statsEl = document.getElementById('role-stats');
    if (!statsEl) return;
    const counts = { player: 0, non_player: 0, unknown: 0 };
    let humanCount = 0;
    for (const t of state.roleTracks) {
        const role = t.role || 'unknown';
        counts[role] = (counts[role] || 0) + 1;
        if (t.role_source === 'human') humanCount++;
    }
    const showing = state.filteredRoleTracks.length;
    const total = state.roleTracks.length;
    const hasModel = state.roleTracks.some(t => t.role_source === 'model');
    const modelTag = hasModel ? ' [ML]' : '';
    statsEl.textContent = `${counts.player}P / ${counts.non_player}NP / ${counts.unknown}? (${showing}/${total}) | ${humanCount} human labels${modelTag}`;
}

async function selectRoleTrack(idx) {
    state.selectedRoleTrackIdx = idx;
    state.trackBboxes = [];
    clearBbox();
    renderRoleTracks();

    const track = state.filteredRoleTracks[idx];
    if (!track) return;

    // Seek video to this track's start
    const videoEl = document.getElementById('review-video');
    if (videoEl) {
        videoEl.currentTime = track.start_frame / state.reviewVideoFps;
    }

    // Load bboxes for the track's full range
    try {
        state.trackBboxes = await api(
            `/videos/${state.reviewVideoId}/tracks/${track.id}/bboxes?start_frame=${track.start_frame}&end_frame=${track.end_frame}`
        );
        drawBbox(track.start_frame);
    } catch { /* ignore */ }
}

async function setTrackRole(trackId, role, event) {
    if (event) event.stopPropagation();
    const currentIdx = state.selectedRoleTrackIdx;

    await api(`/videos/${state.reviewVideoId}/tracks/${trackId}/role`, {
        method: 'PATCH',
        body: JSON.stringify({ role }),
    });

    // Update locally
    const track = state.roleTracks.find(t => t.id === trackId);
    if (track) {
        track.role = role;
        track.role_source = 'human';
    }

    applyRoleFilter();
    renderRoleTracks();
    renderRoleStats();

    // Advance to next track (index may shift after filter re-apply)
    const nextIdx = Math.min(currentIdx, state.filteredRoleTracks.length - 1);
    if (nextIdx >= 0) {
        selectRoleTrack(nextIdx);
        scrollSegmentIntoView();
    }
}

async function selectTrack(idx) {
    state.selectedTrackIdx = idx;
    state.trackBboxes = [];
    clearBbox();
    renderSegments();

    const track = state.phaseTracks[idx];
    if (!track) return;

    // Use the first segment as the anchor for API calls
    const anchorSeg = track.segments[0];
    state.phaseSegmentId = anchorSeg.id;
    state.phaseTrackSegments = track.segments.map(s => s.id);
    state.phaseBoundaries = {};
    state.phasePredicted = false;

    // Seek video to track start
    const videoEl = document.getElementById('review-video');
    if (anchorSeg.start_time != null) {
        videoEl.currentTime = anchorSeg.start_time;
    }

    try {
        // Load full track frame range
        state.phaseFrames = await api(`/tracks/${track.track_id}/frames`);
        // Load bboxes for extended range
        if (state.phaseFrames.length > 0) {
            const extStart = state.phaseFrames[0].frame_number;
            const extEnd = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
            state.trackBboxes = await api(
                `/videos/${state.reviewVideoId}/tracks/${anchorSeg.track_id}/bboxes?start_frame=${extStart}&end_frame=${extEnd}`
            );
            drawBbox(Math.round((anchorSeg.start_time || 0) * state.reviewVideoFps));
        }
        // Load existing phase labels merged across all segments on this track
        const existing = await api(`/segments/${anchorSeg.id}/track-phases`);
        if (existing.length > 0) {
            // Check if these are predictions (no human_label) or human annotations
            state.phasePredicted = existing.every(p => p.human_label == null);
        }
        for (const p of existing) {
            state.phaseBoundaries[p.phase] = p.start_frame;
        }
    } catch { /* ignore */ }

    showPhaseAnnotationPanel();
    renderPhaseTimeline();
}

// ─── Phase Annotation ─────────────────────────────────────────────

function showPhaseAnnotationPanel() {
    document.getElementById('phase-annotation-panel').style.display = '';
    updatePhaseFrameRange();
    // Show/hide prediction indicator and accept button
    const acceptBtn = document.getElementById('accept-phases-btn');
    const predIndicator = document.getElementById('phase-prediction-indicator');
    if (acceptBtn) acceptBtn.style.display = state.phasePredicted ? '' : 'none';
    if (predIndicator) predIndicator.style.display = state.phasePredicted ? '' : 'none';
}

function updatePhaseFrameRange() {
    const el = document.getElementById('phase-frame-range');
    if (!el || state.phaseFrames.length === 0) return;
    const min = state.phaseFrames[0].frame_number;
    const max = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
    el.textContent = `Frames ${min}–${max} (${state.phaseFrames.length})`;
}

async function extendPhaseRange(direction) {
    const track = state.phaseTracks[state.selectedTrackIdx];
    if (!track || state.phaseFrames.length === 0) return;
    const EXTEND_BY = 30;
    const currentMin = state.phaseFrames[0].frame_number;
    const currentMax = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
    const prevCount = state.phaseFrames.length;

    let newStart, newEnd;
    if (direction === 'start') {
        newStart = Math.max(0, currentMin - EXTEND_BY);
        newEnd = currentMax;
    } else {
        newStart = currentMin;
        newEnd = currentMax + EXTEND_BY;
    }

    try {
        state.phaseFrames = await api(
            `/tracks/${track.track_id}/frames?start_frame=${newStart}&end_frame=${newEnd}`
        );
        // Also extend bboxes
        if (state.phaseFrames.length > 0) {
            const extStart = state.phaseFrames[0].frame_number;
            const extEnd = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
            state.trackBboxes = await api(
                `/videos/${state.reviewVideoId}/tracks/${track.track_id}/bboxes?start_frame=${extStart}&end_frame=${extEnd}`
            );
        }
        renderPhaseTimeline();
        updatePhaseFrameRange();
        // Show feedback if no new frames were added
        if (state.phaseFrames.length === prevCount) {
            const el = document.getElementById('phase-frame-range');
            if (el) el.textContent += ' (no more frames)';
        }
    } catch { /* ignore */ }
}

function renderPhaseTimeline() {
    const timeline = document.getElementById('phase-timeline');
    if (!timeline || state.phaseFrames.length === 0) return;

    // Clear existing phase bars (keep playhead)
    timeline.querySelectorAll('.phase-region').forEach(el => el.remove());

    const frames = state.phaseFrames;
    const minFrame = frames[0].frame_number;
    const maxFrame = frames[frames.length - 1].frame_number;
    const range = maxFrame - minFrame || 1;

    // Draw phase regions based on boundaries
    const sortedPhases = PHASE_NAMES.filter(p => state.phaseBoundaries[p] != null);
    for (let i = 0; i < sortedPhases.length; i++) {
        const phase = sortedPhases[i];
        const startFrame = state.phaseBoundaries[phase];
        // End is either the next phase's start or the end of the last phase
        let endFrame;
        if (i + 1 < sortedPhases.length) {
            endFrame = state.phaseBoundaries[sortedPhases[i + 1]];
        } else {
            // Extend to max frame or 30 frames after start
            endFrame = Math.min(startFrame + 30, maxFrame);
        }

        const left = ((startFrame - minFrame) / range) * 100;
        const width = ((endFrame - startFrame) / range) * 100;

        const region = document.createElement('div');
        region.className = 'phase-region' + (state.phasePredicted ? ' phase-predicted' : '');
        region.style.left = left + '%';
        region.style.width = Math.max(width, 0.5) + '%';
        region.style.background = PHASE_COLORS[phase];
        region.title = `${phase}: frame ${startFrame}-${endFrame}${state.phasePredicted ? ' (predicted)' : ''}`;
        timeline.appendChild(region);
    }

    // Show segment boundaries
    const seg = state.segments[state.selectedSegmentIdx];
    if (seg) {
        const segLeft = ((seg.start_frame - minFrame) / range) * 100;
        const segWidth = ((seg.end_frame - seg.start_frame) / range) * 100;
        let marker = timeline.querySelector('.segment-marker');
        if (!marker) {
            marker = document.createElement('div');
            marker.className = 'segment-marker';
            timeline.appendChild(marker);
        }
        marker.style.left = segLeft + '%';
        marker.style.width = segWidth + '%';
    }
}

function updatePhasePlayhead() {
    const playhead = document.getElementById('phase-playhead');
    if (!playhead || state.phaseFrames.length === 0) return;

    const videoEl = document.getElementById('review-video');
    const currentFrame = Math.round(videoEl.currentTime * state.reviewVideoFps);
    const minFrame = state.phaseFrames[0].frame_number;
    const maxFrame = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
    const range = maxFrame - minFrame || 1;

    const pct = ((currentFrame - minFrame) / range) * 100;
    playhead.style.left = Math.max(0, Math.min(100, pct)) + '%';
}

function setPhaseBoundary(phase) {
    const videoEl = document.getElementById('review-video');
    const currentFrame = Math.round(videoEl.currentTime * state.reviewVideoFps);
    state.phaseBoundaries[phase] = currentFrame;
    renderPhaseTimeline();
}

async function savePhases() {
    if (!state.phaseSegmentId) return;

    // Build phase list from boundaries
    const phases = [];
    const sorted = PHASE_NAMES
        .filter(p => state.phaseBoundaries[p] != null)
        .sort((a, b) => state.phaseBoundaries[a] - state.phaseBoundaries[b]);
    if (sorted.length === 0) {
        alert('Set at least one phase boundary before saving.');
        return;
    }

    for (let i = 0; i < sorted.length; i++) {
        const phase = sorted[i];
        const startFrame = state.phaseBoundaries[phase];
        let endFrame;
        if (i + 1 < sorted.length) {
            endFrame = state.phaseBoundaries[sorted[i + 1]] - 1;
        } else {
            endFrame = startFrame + 30; // default end
        }
        phases.push({ phase, start_frame: startFrame, end_frame: Math.max(endFrame, startFrame) });
    }

    try {
        const saved = await api(`/segments/${state.phaseSegmentId}/phases`, {
            method: 'PUT',
            body: JSON.stringify({ phases }),
        });
        // Update track annotation status — all segments on this track are now annotated
        if (state.selectedTrackIdx >= 0 && state.phaseTracks[state.selectedTrackIdx]) {
            const track = state.phaseTracks[state.selectedTrackIdx];
            track.segments.forEach(s => { s._hasPhases = true; });
            track.annotatedCount = track.totalCount;
        }
        // Reload phase boundaries from the merged track-level response
        state.phasePredicted = false;
        state.phaseBoundaries = {};
        if (saved && saved.length > 0) {
            for (const p of saved) {
                state.phaseBoundaries[p.phase] = p.start_frame;
            }
        }
        renderSegments();
        renderPhaseTimeline();
        // Show inline confirmation
        showSaveConfirmation();
    } catch (e) {
        alert('Save failed: ' + e.message);
    }
}

function showSaveConfirmation() {
    const btn = document.getElementById('save-phases-btn');
    if (!btn) return;
    let toast = document.getElementById('phase-save-toast');
    if (!toast) {
        toast = document.createElement('span');
        toast.id = 'phase-save-toast';
        toast.style.cssText = 'color: var(--success); font-weight: 600; font-size: 13px; margin-left: 8px; transition: opacity 0.3s;';
        btn.parentNode.insertBefore(toast, btn.nextSibling);
    }
    toast.textContent = 'Saved!';
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 2000);
}

async function acceptPhases() {
    // Accept predictions as human labels by saving them via the normal save flow
    if (!state.phasePredicted || !state.phaseSegmentId) return;
    await savePhases();
    state.phasePredicted = false;
    showPhaseAnnotationPanel();
    renderPhaseTimeline();
}

async function clearPhases() {
    if (!state.phaseSegmentId) return;
    state.phaseBoundaries = {};
    renderPhaseTimeline();
    try {
        await api(`/segments/${state.phaseSegmentId}/phases?propagate=true`, { method: 'DELETE' });
        // Update track annotation status
        if (state.selectedTrackIdx >= 0 && state.phaseTracks[state.selectedTrackIdx]) {
            const track = state.phaseTracks[state.selectedTrackIdx];
            track.segments.forEach(s => { s._hasPhases = false; });
            track.annotatedCount = 0;
        }
        renderSegments();
    } catch { /* ignore */ }
}

// Phase timeline click to seek
document.getElementById('phase-timeline')?.addEventListener('click', e => {
    if (state.phaseFrames.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const minFrame = state.phaseFrames[0].frame_number;
    const maxFrame = state.phaseFrames[state.phaseFrames.length - 1].frame_number;
    const frame = Math.round(minFrame + pct * (maxFrame - minFrame));
    const videoEl = document.getElementById('review-video');
    videoEl.currentTime = frame / state.reviewVideoFps;
});

// Wire up save/clear buttons
document.getElementById('save-phases-btn')?.addEventListener('click', savePhases);
document.getElementById('clear-phases-btn')?.addEventListener('click', clearPhases);
document.getElementById('extend-start-btn')?.addEventListener('click', () => extendPhaseRange('start'));
document.getElementById('extend-end-btn')?.addEventListener('click', () => extendPhaseRange('end'));
document.getElementById('accept-phases-btn')?.addEventListener('click', acceptPhases);

// Update playhead during video playback
const _reviewVideo = document.getElementById('review-video');
if (_reviewVideo) {
    _reviewVideo.addEventListener('timeupdate', () => {
        if (state.reviewMode === 'phase') updatePhasePlayhead();
    });
}

// Role filter buttons
document.getElementById('role-filter-buttons').addEventListener('click', e => {
    if (e.target.tagName !== 'BUTTON') return;
    document.querySelectorAll('#role-filter-buttons button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    state.roleFilter = e.target.dataset.roleFilter;
    applyRoleFilter();
    renderRoleTracks();
});

// Reclassify tracks button (uses ML model if available, else heuristic)
document.getElementById('reclassify-btn').addEventListener('click', async () => {
    if (!state.reviewVideoId) return;
    try {
        const result = await api(`/videos/${state.reviewVideoId}/tracks/reclassify`, { method: 'POST' });
        const method = result.method === 'ml' ? ' (ML model)' : ' (heuristic)';
        loadRoleTracks();
    } catch (e) {
        alert('Reclassify failed: ' + e.message);
    }
});

// Train role model button
document.getElementById('train-role-btn').addEventListener('click', async () => {
    const btn = document.getElementById('train-role-btn');
    btn.disabled = true;
    btn.textContent = 'Training...';
    try {
        const run = await api('/training/start', {
            method: 'POST',
            body: JSON.stringify({
                task_type: 'role_classification',
                epochs: 1,
                learning_rate: 0.001,
                batch_size: 16,
                dropout: 0.0,
                lstm_units: [64, 32],
            }),
        });
        // Poll for completion (role training is fast)
        const pollInterval = setInterval(async () => {
            try {
                const status = await api(`/training/runs/${run.id}`);
                if (status.status === 'completed') {
                    clearInterval(pollInterval);
                    btn.disabled = false;
                    btn.textContent = 'Train Role Model';
                    const f1 = status.test_f1 != null ? ` CV F1: ${(status.test_f1 * 100).toFixed(1)}%` : '';
                    alert(`Role model trained!${f1}`);
                } else if (status.status === 'failed') {
                    clearInterval(pollInterval);
                    btn.disabled = false;
                    btn.textContent = 'Train Role Model';
                    alert('Training failed: ' + (status.notes || 'Not enough labeled tracks (need 20+)'));
                }
            } catch {
                clearInterval(pollInterval);
                btn.disabled = false;
                btn.textContent = 'Train Role Model';
            }
        }, 1000);
    } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Train Role Model';
        alert('Failed: ' + e.message);
    }
});

// Segment filters
document.getElementById('segment-filters').addEventListener('click', e => {
    if (e.target.tagName !== 'BUTTON') return;
    document.querySelectorAll('#segment-filters button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    state.segmentFilter = e.target.dataset.filter;
    state.selectedSegmentIdx = -1;
    loadSegments();
});

// Keyboard shortcuts for review
document.addEventListener('keydown', e => {
    if (state.currentView !== 'review') return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    const seg = state.segments[state.selectedSegmentIdx];

    if (state.reviewMode === 'role') {
        // Role ID mode shortcuts
        const track = state.filteredRoleTracks[state.selectedRoleTrackIdx];
        switch (e.key) {
            case 'p':
            case 'P':
                if (track) setTrackRole(track.id, 'player');
                break;
            case 'x':
            case 'X':
                if (track) setTrackRole(track.id, 'non_player');
                break;
            case 'ArrowDown':
                e.preventDefault();
                if (state.selectedRoleTrackIdx < state.filteredRoleTracks.length - 1) {
                    selectRoleTrack(state.selectedRoleTrackIdx + 1);
                    scrollSegmentIntoView();
                }
                break;
            case 'ArrowUp':
                e.preventDefault();
                if (state.selectedRoleTrackIdx > 0) {
                    selectRoleTrack(state.selectedRoleTrackIdx - 1);
                    scrollSegmentIntoView();
                }
                break;
        }
        return;
    }

    if (state.reviewMode === 'phase') {
        // Phase mode shortcuts
        const videoEl = document.getElementById('review-video');
        switch (e.key) {
            case '1': setPhaseBoundary('approach'); break;
            case '2': setPhaseBoundary('jump'); break;
            case '3': setPhaseBoundary('swing'); break;
            case '4': setPhaseBoundary('land'); break;
            case 'ArrowLeft':
                e.preventDefault();
                if (videoEl) videoEl.currentTime = Math.max(0, videoEl.currentTime - 1 / state.reviewVideoFps);
                break;
            case 'ArrowRight':
                e.preventDefault();
                if (videoEl) videoEl.currentTime += 1 / state.reviewVideoFps;
                break;
            case 'Enter':
                e.preventDefault();
                savePhases();
                break;
            case 'ArrowDown':
                e.preventDefault();
                if (state.selectedTrackIdx < state.phaseTracks.length - 1) {
                    selectTrack(state.selectedTrackIdx + 1);
                    scrollTrackIntoView();
                }
                break;
            case 'ArrowUp':
                e.preventDefault();
                if (state.selectedTrackIdx > 0) {
                    selectTrack(state.selectedTrackIdx - 1);
                    scrollTrackIntoView();
                }
                break;
        }
        return;
    }

    // Spike mode shortcuts
    switch (e.key) {
        case 's':
        case 'S':
            if (seg) labelSegment(seg.id, 1);
            break;
        case 'n':
        case 'N':
            if (seg) labelSegment(seg.id, 0);
            break;
        case 'ArrowDown':
            e.preventDefault();
            if (state.selectedSegmentIdx < state.segments.length - 1) {
                selectSegment(state.selectedSegmentIdx + 1);
                scrollSegmentIntoView();
            }
            break;
        case 'ArrowUp':
            e.preventDefault();
            if (state.selectedSegmentIdx > 0) {
                selectSegment(state.selectedSegmentIdx - 1);
                scrollSegmentIntoView();
            }
            break;
    }
});

function scrollSegmentIntoView() {
    const card = document.querySelector(`.segment-card[data-idx="${state.selectedSegmentIdx}"]`);
    if (card) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function scrollTrackIntoView() {
    const card = document.querySelector(`.segment-card[data-idx="${state.selectedTrackIdx}"]`);
    if (card) card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ─── View 3: Training Dashboard ──────────────────────────────────

// Task type selector reloads training runs
document.getElementById('training-task-type').addEventListener('change', () => {
    loadTrainingRuns();
    loadLabelStats();
});

document.getElementById('training-form').addEventListener('submit', async e => {
    e.preventDefault();
    const form = e.target;
    const config = {
        task_type: form.task_type.value,
        epochs: parseInt(form.epochs.value),
        learning_rate: parseFloat(form.learning_rate.value),
        batch_size: parseInt(form.batch_size.value),
        dropout: parseFloat(form.dropout.value),
        class_weight_positive: parseFloat(form.class_weight_positive.value),
        balance_by_group: document.getElementById('balance-by-group').checked,
        lstm_units: [64, 32],
    };

    try {
        const run = await api('/training/start', {
            method: 'POST',
            body: JSON.stringify(config),
        });
        document.getElementById('training-status').textContent = `Training run #${run.id} started...`;
        pollTrainingRun(run.id);
    } catch (e) {
        document.getElementById('training-status').textContent = `Error: ${e.message}`;
    }
});

function pollTrainingRun(runId) {
    const interval = setInterval(async () => {
        try {
            const run = await api(`/training/runs/${runId}`);
            const statusEl = document.getElementById('training-status');

            if (run.status === 'running') {
                const loss = run.val_loss != null ? `val_loss: ${run.val_loss.toFixed(4)}` : '';
                statusEl.textContent = `Training... Epoch ${run.best_epoch || '?'}/${run.epochs} ${loss}`;
            } else if (run.status === 'completed') {
                statusEl.textContent = `Training complete! Accuracy: ${(run.test_accuracy * 100).toFixed(1)}% F1: ${(run.test_f1 * 100).toFixed(1)}%`;
                clearInterval(interval);
                loadTrainingRuns();
            } else {
                statusEl.textContent = `Training ${run.status}`;
                clearInterval(interval);
                loadTrainingRuns();
            }
        } catch {
            clearInterval(interval);
        }
    }, 3000);
}

async function loadTrainingRuns() {
    try {
        const taskType = document.getElementById('training-task-type')?.value || 'spike_detection';
        const runs = await api(`/training/runs?task_type=${taskType}`);
        const tbody = document.getElementById('training-table-body');
        tbody.innerHTML = '';

        // Find best F1 among completed runs
        const completedRuns = runs.filter(r => r.status === 'completed' && r.test_f1 != null);
        const bestF1 = completedRuns.length > 0 ? Math.max(...completedRuns.map(r => r.test_f1)) : -1;

        for (const r of runs) {
            const tr = document.createElement('tr');
            const acc = r.test_accuracy != null ? `${(r.test_accuracy * 100).toFixed(1)}%` : '-';
            const prec = r.test_precision != null ? `${(r.test_precision * 100).toFixed(1)}%` : '-';
            const recall = r.test_recall != null ? `${(r.test_recall * 100).toFixed(1)}%` : '-';
            const f1 = r.test_f1 != null ? `${(r.test_f1 * 100).toFixed(1)}%` : '-';
            const auc = r.test_auc != null ? `${(r.test_auc * 100).toFixed(1)}%` : '-';
            const loss = r.val_loss != null ? r.val_loss.toFixed(4) : '-';
            const samples = r.train_count != null ? `${r.train_count}/${r.val_count}/${r.test_count}` : '-';
            const isBest = r.test_f1 != null && r.test_f1 === bestF1;

            if (isBest) tr.classList.add('best-run');

            let inferBtn = '';
            if (taskType === 'phase_classification') {
                inferBtn = `<button class="btn btn-sm btn-outline" onclick="rerunPhaseInference(${r.id})">Phase Infer</button>`;
            } else if (taskType === 'spike_detection') {
                inferBtn = `<button class="btn btn-sm btn-outline" onclick="rerunInference(${r.id})">Re-infer</button>`;
            }
            // role_classification runs don't have a re-infer action (use Reclassify in Role ID tab instead)

            const notesDisplay = r.notes ? esc(r.notes) : '<span style="color:var(--text-muted);">—</span>';

            tr.innerHTML = `
                <td>#${r.id}${isBest ? ' <span style="color:var(--success); font-size:11px;">best</span>' : ''}</td>
                <td class="run-notes-cell" data-run-id="${r.id}" style="max-width:140px; cursor:pointer;" title="Click to edit">${notesDisplay}</td>
                <td><span class="badge badge-${r.status}">${r.status}</span></td>
                <td>${samples}</td>
                <td>${acc}</td>
                <td>${prec}</td>
                <td>${recall}</td>
                <td>${f1}</td>
                <td>${auc}</td>
                <td>${loss}</td>
                <td>
                    ${r.status === 'completed' ? inferBtn : ''}
                    <button class="btn btn-sm btn-danger" onclick="deleteTrainingRun(${r.id})" style="margin-left:4px;">Del</button>
                </td>
            `;
            tbody.appendChild(tr);
        }

        // Wire up inline notes editing
        tbody.querySelectorAll('.run-notes-cell').forEach(cell => {
            cell.addEventListener('click', () => {
                if (cell.querySelector('input')) return; // already editing
                const runId = cell.dataset.runId;
                const current = cell.textContent.trim() === '—' ? '' : cell.textContent.trim();
                const input = document.createElement('input');
                input.type = 'text';
                input.value = current;
                input.style.cssText = 'width:100%; padding:2px 4px; background:var(--bg); color:var(--text); border:1px solid var(--primary); border-radius:3px; font-size:12px;';
                cell.innerHTML = '';
                cell.appendChild(input);
                input.focus();

                const save = async () => {
                    const val = input.value.trim();
                    try {
                        await api(`/training/runs/${runId}/notes`, {
                            method: 'PATCH',
                            body: JSON.stringify({ notes: val || null }),
                        });
                    } catch { /* ignore */ }
                    cell.innerHTML = val ? esc(val) : '<span style="color:var(--text-muted);">—</span>';
                };
                input.addEventListener('blur', save);
                input.addEventListener('keydown', e => { if (e.key === 'Enter') input.blur(); });
            });
        });

        renderMetricsChart(runs);
    } catch { /* ignore on initial load */ }
}

async function deleteTrainingRun(runId) {
    if (!confirm(`Delete training run #${runId}? This will remove the checkpoint and clear predictions from this run.`)) return;
    try {
        await api(`/training/runs/${runId}`, { method: 'DELETE' });
        loadTrainingRuns();
    } catch (e) {
        alert('Failed to delete run: ' + e.message);
    }
}

let _metricsChart = null;

function renderMetricsChart(runs) {
    const canvas = document.getElementById('metrics-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    // Filter to completed runs, sort by id (chronological)
    const completed = runs
        .filter(r => r.status === 'completed' && r.test_f1 != null)
        .sort((a, b) => a.id - b.id);

    if (completed.length === 0) return;

    const labels = completed.map(r => {
        const total = (r.train_count || 0) + (r.val_count || 0) + (r.test_count || 0);
        return `#${r.id} (${total > 1000 ? (total / 1000).toFixed(1) + 'k' : total})`;
    });

    // Compute dynamic y-axis min from data
    const allMetricValues = completed.flatMap(r =>
        [r.test_f1, r.test_precision, r.test_recall, r.test_accuracy]
        .filter(v => v != null)
        .map(v => v * 100)
    );
    const dataMin = allMetricValues.length > 0 ? Math.min(...allMetricValues) : 60;
    const yMin = Math.max(0, Math.floor(dataMin / 10) * 10 - 10);

    // Destroy previous chart to prevent canvas reuse errors
    if (_metricsChart) {
        _metricsChart.destroy();
        _metricsChart = null;
    }

    _metricsChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'F1',
                    data: completed.map(r => (r.test_f1 * 100).toFixed(1)),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 4,
                },
                {
                    label: 'Precision',
                    data: completed.map(r => (r.test_precision * 100).toFixed(1)),
                    borderColor: '#22c55e',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Recall',
                    data: completed.map(r => (r.test_recall * 100).toFixed(1)),
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 3,
                },
                {
                    label: 'Accuracy',
                    data: completed.map(r => (r.test_accuracy * 100).toFixed(1)),
                    borderColor: '#6b7280',
                    borderWidth: 1,
                    borderDash: [4, 4],
                    tension: 0.3,
                    pointRadius: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#e4e4e7', font: { size: 12 } },
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%`,
                    },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#71717a', font: { size: 11 } },
                    grid: { color: 'rgba(42, 46, 58, 0.5)' },
                },
                y: {
                    min: yMin,
                    max: 100,
                    ticks: {
                        color: '#71717a',
                        font: { size: 11 },
                        callback: v => v + '%',
                    },
                    grid: { color: 'rgba(42, 46, 58, 0.5)' },
                },
            },
        },
    });
}

async function showInferenceVideoPicker(runId, mode) {
    const videos = await api('/videos');
    const eligible = videos.filter(v => v.status === 'processed' || v.status === 'predicted');

    if (eligible.length === 0) {
        alert('No videos available for inference.');
        return;
    }

    // Remove existing picker if any
    document.getElementById('inference-picker-overlay')?.remove();

    const overlay = document.createElement('div');
    overlay.id = 'inference-picker-overlay';
    overlay.style.cssText = 'position:fixed; inset:0; background:rgba(0,0,0,0.6); display:flex; align-items:center; justify-content:center; z-index:1000;';

    const dialog = document.createElement('div');
    dialog.style.cssText = 'background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:20px; min-width:320px; max-width:480px;';

    const title = mode === 'phase' ? 'Run Phase Inference' : 'Run Spike Inference';
    let html = `<h3 style="margin:0 0 12px; font-size:14px;">${title}</h3>`;
    html += `<p style="font-size:13px; color:var(--text-muted); margin:0 0 12px;">Select video(s) to run inference on with run #${runId}:</p>`;
    html += `<div style="margin-bottom:8px;"><button class="btn btn-sm btn-outline" id="infer-toggle-all">Deselect All</button></div>`;
    html += `<div id="infer-video-list" style="display:flex; flex-direction:column; gap:6px; max-height:300px; overflow-y:auto;">`;
    for (const v of eligible) {
        html += `<label style="display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer;">
            <input type="checkbox" value="${v.id}" checked style="accent-color:var(--primary);">
            <span>${esc(v.filename)}</span>
            <span style="color:var(--text-muted); font-size:11px;">(${v.segment_count} seg)</span>
        </label>`;
    }
    html += `</div>`;
    html += `<div style="display:flex; gap:8px; margin-top:16px; justify-content:flex-end;">`;
    html += `<button class="btn btn-sm btn-outline" id="infer-cancel-btn">Cancel</button>`;
    html += `<button class="btn btn-sm btn-primary" id="infer-run-btn">Run</button>`;
    html += `</div>`;

    dialog.innerHTML = html;
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Close on overlay click (not dialog)
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    document.getElementById('infer-cancel-btn').addEventListener('click', () => overlay.remove());

    // Toggle all checkboxes
    document.getElementById('infer-toggle-all').addEventListener('click', () => {
        const checkboxes = dialog.querySelectorAll('#infer-video-list input[type="checkbox"]');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        checkboxes.forEach(cb => cb.checked = !allChecked);
        document.getElementById('infer-toggle-all').textContent = allChecked ? 'Select All' : 'Deselect All';
    });

    document.getElementById('infer-run-btn').addEventListener('click', async () => {
        const checked = dialog.querySelectorAll('input[type="checkbox"]:checked');
        const videoIds = Array.from(checked).map(cb => cb.value);
        overlay.remove();

        if (videoIds.length === 0) {
            alert('No videos selected.');
            return;
        }

        if (mode === 'phase') {
            await runPhaseInferenceOnVideos(runId, videoIds);
        } else {
            await runSpikeInferenceOnVideos(runId, videoIds);
        }
    });
}

async function runSpikeInferenceOnVideos(runId, videoIds) {
    let count = 0;
    for (const vid of videoIds) {
        try {
            await api('/inference/run', {
                method: 'POST',
                body: JSON.stringify({ video_id: vid, training_run_id: runId }),
            });
            count++;
        } catch (e) {
            if (e.message.includes('already running')) {
                alert(`Inference started for ${count} video(s). Worker is busy — remaining videos will need to be run separately.`);
                return;
            }
        }
    }
    alert(`Inference started for ${count} video(s). Predictions will update shortly.`);
}

async function runPhaseInferenceOnVideos(runId, videoIds) {
    let count = 0;
    for (const vid of videoIds) {
        try {
            await api('/inference/phase-run', {
                method: 'POST',
                body: JSON.stringify({ video_id: vid, training_run_id: runId }),
            });
            count++;
        } catch (e) {
            if (e.message.includes('already running')) {
                alert(`Phase inference started for ${count} video(s). Worker is busy — remaining videos will need to be run separately.`);
                return;
            }
        }
    }
    alert(`Phase inference started for ${count} video(s). Predictions will appear in Phase Annotation mode.`);
}

async function rerunInference(runId) {
    showInferenceVideoPicker(runId, 'spike');
}

async function rerunPhaseInference(runId) {
    showInferenceVideoPicker(runId, 'phase');
}

async function loadLabelStats() {
    try {
        const stats = await api('/inference/stats');
        document.getElementById('label-stats').innerHTML = `
            <span>Total segments: <strong>${stats.total}</strong></span> &nbsp;|&nbsp;
            <span>Labeled: <strong>${stats.labeled}</strong></span> &nbsp;|&nbsp;
            <span style="color:var(--spike);">Spike: <strong>${stats.spike}</strong></span> &nbsp;|&nbsp;
            <span>Non-spike: <strong>${stats.non_spike}</strong></span> &nbsp;|&nbsp;
            <span>Unlabeled: <strong>${stats.unlabeled}</strong></span>
        `;
    } catch { /* ignore on initial load */ }
}

// ─── Group Metrics ────────────────────────────────────────────────

const _groupCalibrationCharts = [];

async function loadGroupMetricsRunSelector() {
    const select = document.getElementById('group-metrics-run-select');
    if (!select) return;
    try {
        const runs = await api('/training/runs?task_type=spike_detection');
        const completed = runs.filter(r => r.status === 'completed');
        select.innerHTML = '<option value="">Select a training run...</option>';
        for (const r of completed) {
            const f1 = r.test_f1 != null ? ` (F1: ${(r.test_f1 * 100).toFixed(1)}%)` : '';
            select.innerHTML += `<option value="${r.id}">#${r.id}${f1}</option>`;
        }
    } catch { /* ignore */ }
}

document.getElementById('group-metrics-run-select')?.addEventListener('change', e => {
    const runId = e.target.value;
    if (runId) loadGroupMetrics(parseInt(runId));
});

async function loadGroupMetrics(runId) {
    const container = document.getElementById('group-metrics-container');
    if (!container) return;
    container.innerHTML = '<span style="color:var(--text-muted);">Loading...</span>';

    // Destroy old calibration charts
    _groupCalibrationCharts.forEach(c => c.destroy());
    _groupCalibrationCharts.length = 0;

    try {
        const data = await api(`/training/group-metrics?training_run_id=${runId}`);
        if (!data.groups || data.groups.length === 0) {
            container.innerHTML = '<span style="color:var(--text-muted);">No segments with both predictions and labels found for this run.</span>';
            return;
        }
        container.innerHTML = '';
        for (const group of data.groups) {
            container.appendChild(renderGroupCard(group));
        }
    } catch (e) {
        container.innerHTML = `<span style="color:var(--danger, #ef4444);">Error: ${esc(e.message)}</span>`;
    }
}

function renderGroupCard(group) {
    const card = document.createElement('div');
    card.className = 'group-card';

    const isOverall = group.group_name === 'overall';
    const titleStyle = isOverall ? 'font-weight:700;' : '';

    let html = `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <span style="font-size:14px; ${titleStyle}">${esc(group.group_name.replace(/_/g, ' '))}${isOverall ? ' (all groups)' : ''}</span>
        <span style="font-size:12px; color:var(--text-muted);">${group.total} samples (${group.spike_count}S / ${group.non_spike_count}NS)</span>
    </div>`;

    // Metrics row
    html += `<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
        <span class="metric-badge metric-good">${(group.precision * 100).toFixed(1)}% <small>Prec</small></span>
        <span class="metric-badge metric-good">${(group.recall * 100).toFixed(1)}% <small>Recall</small></span>
        <span class="metric-badge metric-good">${(group.f1 * 100).toFixed(1)}% <small>F1</small></span>
        <span class="metric-badge metric-bad">${(group.fpr * 100).toFixed(1)}% <small>FPR</small></span>
        <span class="metric-badge metric-bad">${(group.fnr * 100).toFixed(1)}% <small>FNR</small></span>
    </div>`;

    // Confusion matrix
    html += `<div style="display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start;">`;
    html += `<div>
        <div style="font-size:11px; color:var(--text-muted); margin-bottom:4px;">Confusion Matrix</div>
        <div class="confusion-matrix">
            <div class="cm-header"></div>
            <div class="cm-header">Pred S</div>
            <div class="cm-header">Pred NS</div>
            <div class="cm-header">True S</div>
            <div class="cm-cell cm-tp">${group.tp}</div>
            <div class="cm-cell cm-fn">${group.fn}</div>
            <div class="cm-header">True NS</div>
            <div class="cm-cell cm-fp">${group.fp}</div>
            <div class="cm-cell cm-tn">${group.tn}</div>
        </div>
    </div>`;

    // Calibration chart placeholder
    const chartId = `cal-chart-${group.group_name.replace(/\W/g, '_')}-${Date.now()}`;
    html += `<div style="flex:1; min-width:200px; max-width:320px;">
        <div style="font-size:11px; color:var(--text-muted); margin-bottom:4px;">Calibration Curve</div>
        <div style="height:160px;"><canvas id="${chartId}"></canvas></div>
    </div>`;

    html += `</div>`;

    // Per-video breakdown (collapsible)
    if (group.per_video && group.per_video.length > 0) {
        html += `<details style="margin-top:12px;">
            <summary style="cursor:pointer; font-size:11px; color:var(--text-muted);">Per-Video F1 (mean: ${(group.f1_mean * 100).toFixed(1)}% &plusmn; ${(group.f1_std * 100).toFixed(1)}%)</summary>
            <table style="width:100%; font-size:12px; margin-top:4px;">
                <thead><tr><th style="text-align:left;">Video</th><th>N</th><th>Prec</th><th>Recall</th><th>F1</th></tr></thead>
                <tbody>`;
        for (const v of group.per_video) {
            html += `<tr>
                <td style="text-align:left; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${esc(v.filename)}</td>
                <td>${v.n_samples}</td>
                <td>${v.precision != null ? (v.precision * 100).toFixed(1) + '%' : '-'}</td>
                <td>${v.recall != null ? (v.recall * 100).toFixed(1) + '%' : '-'}</td>
                <td>${v.f1 != null ? (v.f1 * 100).toFixed(1) + '%' : '-'}</td>
            </tr>`;
        }
        html += `</tbody></table></details>`;
    }

    card.innerHTML = html;

    // Render calibration chart after DOM insertion
    requestAnimationFrame(() => {
        const canvas = document.getElementById(chartId);
        if (canvas && typeof Chart !== 'undefined') {
            const buckets = group.calibration.filter(b => b.count > 0);
            const chart = new Chart(canvas.getContext('2d'), {
                type: 'line',
                data: {
                    labels: buckets.map(b => ((b.bin_start + b.bin_end) / 2 * 100).toFixed(0) + '%'),
                    datasets: [
                        {
                            label: 'Actual positive rate',
                            data: buckets.map(b => b.actual_positive_rate != null ? (b.actual_positive_rate * 100).toFixed(1) : null),
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59,130,246,0.1)',
                            borderWidth: 2,
                            pointRadius: 3,
                            tension: 0.2,
                        },
                        {
                            label: 'Perfect calibration',
                            data: buckets.map(b => ((b.bin_start + b.bin_end) / 2 * 100).toFixed(1)),
                            borderColor: '#6b7280',
                            borderWidth: 1,
                            borderDash: [4, 4],
                            pointRadius: 0,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => {
                                    const bucket = group.calibration.filter(b => b.count > 0)[ctx.dataIndex];
                                    const n = bucket ? bucket.count : 0;
                                    return `${ctx.dataset.label}: ${ctx.parsed.y}% (n=${n})`;
                                },
                            },
                        },
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Predicted confidence', color: '#71717a', font: { size: 10 } },
                            ticks: { color: '#71717a', font: { size: 10 } },
                            grid: { color: 'rgba(42,46,58,0.5)' },
                        },
                        y: {
                            min: 0, max: 100,
                            title: { display: true, text: 'Actual positive %', color: '#71717a', font: { size: 10 } },
                            ticks: { color: '#71717a', font: { size: 10 }, callback: v => v + '%' },
                            grid: { color: 'rgba(42,46,58,0.5)' },
                        },
                    },
                },
            });
            _groupCalibrationCharts.push(chart);
        }
    });

    return card;
}

// ─── Bbox Overlay ─────────────────────────────────────────────────

function drawBbox(currentFrame) {
    const canvas = document.getElementById('bbox-canvas');
    const videoEl = document.getElementById('review-video');
    if (!canvas || !videoEl) return;

    const ctx = canvas.getContext('2d');

    // Match canvas resolution to video display size
    const rect = videoEl.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    ctx.clearRect(0, 0, rect.width, rect.height);

    if (state.trackBboxes.length === 0) return;

    // Find the closest bbox to the current frame
    let closest = state.trackBboxes[0];
    let minDist = Math.abs(closest.frame - currentFrame);
    for (const b of state.trackBboxes) {
        const dist = Math.abs(b.frame - currentFrame);
        if (dist < minDist) { closest = b; minDist = dist; }
    }

    // Don't draw if too far from any bbox frame (>5 frames away)
    if (minDist > 5) return;

    const [x1, y1, x2, y2] = closest.bbox;

    // Scale from video pixel coords to display coords
    const scaleX = rect.width / state.reviewVideoWidth;
    const scaleY = rect.height / state.reviewVideoHeight;

    const dx = x1 * scaleX;
    const dy = y1 * scaleY;
    const dw = (x2 - x1) * scaleX;
    const dh = (y2 - y1) * scaleY;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.strokeRect(dx, dy, dw, dh);

    // Label
    const label = `Track ${state.segments[state.selectedSegmentIdx]?.track_id ?? ''}`;
    ctx.font = '12px -apple-system, sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = '#3b82f6';
    ctx.fillRect(dx, dy - 18, textWidth + 8, 18);
    ctx.fillStyle = 'white';
    ctx.fillText(label, dx + 4, dy - 5);
}

function clearBbox() {
    const canvas = document.getElementById('bbox-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ─── Utilities ────────────────────────────────────────────────────

function esc(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

// ─── Init ─────────────────────────────────────────────────────────

loadVideos();
