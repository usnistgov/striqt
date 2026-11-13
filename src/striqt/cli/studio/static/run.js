// the Run tab: the output path, the adjustments table, Start/Stop, progress, the
// live plot, the log console, and the operator prompt dialog

import { Plot } from './plot.js';
import { el, busy } from './util.js';

const LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'];
const MAX_LOG_ROWS = 2000;

function pad(n, width) {
    return String(n).padStart(width, '0');
}

function clock(unixSeconds) {
    const d = new Date(unixSeconds * 1000);
    return `${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}:${pad(d.getSeconds(), 2)}`
        + `.${pad(d.getMilliseconds(), 3)}`;
}

function duration(seconds) {
    if (seconds < 60) {
        return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
    }
    const total = Math.round(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h > 0 ? `${h}:${pad(m, 2)}:${pad(s, 2)}` : `${m}:${pad(s, 2)}`;
}

export function init(app) {
    const panel = document.getElementById('tab-run');
    panel.innerHTML = `
        <div class="sidebar run-sidebar">
            <div class="run-heading">
                <span class="run-action-class">&mdash;</span>
                <span class="run-spec-filename"></span>
            </div>
            <div class="run-block run-output">
                <h2 class="panel-heading">output</h2>
                <div class="run-output-row">
                    <code class="run-sink-path"></code>
                    <span class="run-badge button outline"></span>
                </div>
                <label class="run-checkbox" hidden>
                    <input type="checkbox" class="run-overwrite"> overwrite the existing output
                </label>
            </div>
            <div class="control-group run-level">
                <label for="run-log-level">log level</label>
                <select id="run-log-level">
                    ${LEVELS.map(l => `<option value="${l}"${l === 'INFO' ? ' selected' : ''}>${l}</option>`).join('')}
                </select>
            </div>
            <div class="run-block run-adjustments">
                <h2 class="panel-heading">capture adjustments</h2>
                <div class="run-adjustments-body"></div>
                <p class="note run-note">Start confirms these values</p>
            </div>
            <button type="button" class="button run-start" disabled>Start</button>
            <p class="note run-hint"></p>
            <div class="run-block run-progress" hidden>
                <h2 class="panel-heading">progress</h2>
                <div class="run-bar"><div class="run-bar-fill"></div></div>
                <div class="run-progress-counts"></div>
                <div class="run-progress-text"></div>
                <div class="run-elapsed"></div>
                <div class="run-finished"></div>
                <div class="run-failed" hidden>
                    <div class="run-failed-line"></div>
                    <details><summary>traceback</summary><pre class="run-traceback"></pre></details>
                </div>
            </div>
        </div>
        <div class="run-main">
            <div class="run-plot-stage">
                <div class="run-plot"><div class="loading">Waiting for a run...</div></div>
            </div>
            <div class="run-console min-INFO">
                <div class="run-console-toolbar">
                    <span class="run-console-title panel-heading">log</span>
                    <span class="run-console-hidden-note"></span>
                    <button type="button" class="button run-console-clear">clear</button>
                </div>
                <div class="run-console-rows"></div>
            </div>
        </div>
    `;

    const q = (selector) => panel.querySelector(selector);
    const actionClass = q('.run-action-class');
    const specFilename = q('.run-spec-filename');
    const sinkPath = q('.run-sink-path');
    const badge = q('.run-badge');
    const overwriteLabel = q('.run-checkbox');
    const overwriteBox = q('.run-overwrite');
    const levelSelect = q('#run-log-level');
    const adjustmentsBody = q('.run-adjustments-body');
    const startButton = q('.run-start');
    const hint = q('.run-hint');
    const progress = q('.run-progress');
    const barFill = q('.run-bar-fill');
    const counts = q('.run-progress-counts');
    const progressText = q('.run-progress-text');
    const elapsed = q('.run-elapsed');
    const finished = q('.run-finished');
    const failed = q('.run-failed');
    const failedLine = q('.run-failed-line');
    const traceback = q('.run-traceback');
    const consoleBox = q('.run-console');
    const consoleRows = q('.run-console-rows');
    const hiddenNote = q('.run-console-hidden-note');

    const plot = new Plot(q('.run-plot'));
    // the plot draws on every message so it is current when the tab opens;
    // plotly sizes a hidden container to 0, so a resize follows each tab show
    app.on('plot_data', (msg) => plot.draw(msg.data));
    app.onTab('run', () => plot.resize());
    window.addEventListener('resize', () => plot.resize());

    // %% the state

    let state = null;
    let ticker = null;

    app.on('state', (msg) => {
        state = msg;
        renderHeading(msg);
        renderOutput(msg.run);
        renderAdjustments(msg.run.adjustments);
        renderProgress(msg);
        updateButton();
        renderPrompt(msg.prompt);
    });

    app.on('connection', () => updateButton());

    function renderHeading(msg) {
        actionClass.textContent = msg.action_class || '—';
        specFilename.textContent = msg.spec_filename || '';
    }

    function renderOutput(run) {
        badge.className = 'run-badge button outline';
        if (run.sink_path === null) {
            sinkPath.textContent = '';
            sinkPath.hidden = true;
            badge.classList.add('none');
            badge.textContent = 'no output (sink.path is null)';
        } else {
            sinkPath.hidden = false;
            sinkPath.textContent = run.sink_path;
            badge.classList.add(run.exists ? 'exists' : 'new');
            badge.textContent = run.exists ? 'exists' : 'new';
        }
        overwriteLabel.hidden = !run.exists;
        if (!run.exists) {
            overwriteBox.checked = false;
        }
    }

    function renderAdjustments(adjustments) {
        adjustmentsBody.innerHTML = '';
        const fields = Object.keys(adjustments || {});
        if (fields.length === 0) {
            adjustmentsBody.appendChild(el('p', 'placeholder', 'no capture adjustments'));
            return;
        }
        const table = el('table', 'coords-table');
        const tbody = el('tbody');
        for (const field of fields) {
            const row = el('tr');
            row.appendChild(el('td', 'coord-name', field));
            row.appendChild(el('td', 'coord-value', adjustments[field].map(String).join(', ')));
            tbody.appendChild(row);
        }
        table.appendChild(tbody);
        adjustmentsBody.appendChild(table);
    }

    function isRunning() {
        return !!state && busy(state.mode);
    }

    function outputBlocked() {
        return !!state && state.run.exists && !overwriteBox.checked;
    }

    function updateButton() {
        startButton.classList.remove('stop');
        hint.textContent = '';
        if (!state) {
            startButton.textContent = 'Start';
            startButton.disabled = true;
            return;
        }
        if (isRunning()) {
            startButton.textContent = 'Stop';
            startButton.classList.add('stop');
            startButton.disabled = !app.connected();
            if (!app.connected()) {
                hint.textContent = 'disconnected from the server';
            }
            return;
        }
        startButton.textContent = 'Start';
        if (!app.connected()) {
            startButton.disabled = true;
            hint.textContent = 'disconnected from the server';
        } else if (state.mode !== 'preview') {
            startButton.disabled = true;
            hint.textContent = state.mode === 'closed'
                ? 'the acquisition session is closed'
                : `a run starts from the preview (the session is ${state.mode})`;
        } else if (outputBlocked()) {
            startButton.disabled = true;
            hint.textContent = 'the output exists: check overwrite to replace it';
        } else {
            startButton.disabled = false;
        }
    }

    overwriteBox.addEventListener('change', updateButton);

    startButton.addEventListener('click', () => {
        if (isRunning()) {
            app.send({ type: 'run_stop' });
        } else {
            app.send({
                type: 'run_start',
                overwrite: overwriteBox.checked,
                log_level: levelSelect.value,
            });
        }
    });

    levelSelect.addEventListener('change', () => {
        applyLevelFilter();
        if (isRunning()) {
            app.send({ type: 'set_log_level', level: levelSelect.value });
        }
    });

    // %% progress

    function renderProgress(msg) {
        const run = msg.run;
        if (run.count === undefined) {
            progress.hidden = true;
            stopTicker();
            return;
        }
        progress.hidden = false;
        const fraction = run.count > 0 ? Math.min(run.acquired / run.count, 1) : 0;
        barFill.style.width = `${(100 * fraction).toFixed(1)}%`;
        barFill.classList.toggle('error', !!run.error);
        counts.textContent = `${run.acquired} / ${run.count} (${Math.round(100 * fraction)}%)`;

        if (run.capture_index >= 0) {
            const of = run.captures_per_sweep ? ` of ${run.captures_per_sweep}` : '';
            progressText.textContent =
                `capture ${run.capture_index + 1}${of}, sweep ${run.sweep_index + 1}`;
        } else {
            progressText.textContent = msg.mode === 'prompting'
                ? 'waiting for an answer'
                : 'waiting for the first acquisition';
        }

        failed.hidden = !run.error;
        finished.textContent = '';
        if (run.error) {
            const lines = run.error.trim().split('\n');
            failedLine.textContent = lines[lines.length - 1];
            traceback.textContent = run.error;
        } else if (run.finished !== null) {
            finished.textContent = `finished in ${duration(run.finished - run.started)}`;
        }

        tickElapsed();
        if (run.finished === null) {
            startTicker();
        } else {
            stopTicker();
        }
    }

    function tickElapsed() {
        const run = state && state.run;
        if (!run || run.count === undefined) {
            return;
        }
        const end = run.finished === null ? Date.now() / 1000 : run.finished;
        elapsed.textContent = `elapsed ${duration(Math.max(end - run.started, 0))}`;
    }

    function startTicker() {
        if (ticker === null) {
            ticker = setInterval(tickElapsed, 1000);
        }
    }

    function stopTicker() {
        if (ticker !== null) {
            clearInterval(ticker);
            ticker = null;
        }
    }

    // %% the log console

    // the user scrolls up to read; the pane sticks to the bottom until then
    let followTail = true;
    consoleRows.addEventListener('scroll', () => {
        followTail = consoleRows.scrollTop + consoleRows.clientHeight >= consoleRows.scrollHeight - 4;
    });

    // Records below the selected level stay in the DOM and are hidden by the
    // `min-<LEVEL>` class, so lowering the level reveals what already arrived.
    function applyLevelFilter() {
        for (const level of LEVELS) {
            consoleBox.classList.toggle(`min-${level}`, level === levelSelect.value);
        }
        hiddenNote.textContent = levelSelect.value === 'DEBUG' ? '' : `below ${levelSelect.value} hidden`;
        if (followTail) {
            consoleRows.scrollTop = consoleRows.scrollHeight;
        }
    }

    app.on('log_record', (msg) => {
        const row = el('div', `run-log-row level-${LEVELS.includes(msg.level) ? msg.level : 'INFO'}`);
        row.appendChild(el('span', 'log-time', clock(msg.time)));
        row.appendChild(el('span', 'log-level', msg.level.padEnd(7)));
        row.appendChild(el('span', 'log-logger', msg.logger));
        row.appendChild(el('span', 'log-sep', ' • '));
        if (msg.capture_progress) {
            row.appendChild(el('span', 'log-progress', `[${msg.capture_progress}] `));
        }
        row.appendChild(el('span', 'log-message', msg.message));
        if (msg.exception) {
            row.appendChild(el('pre', 'log-exception', msg.exception));
        }
        consoleRows.appendChild(row);
        while (consoleRows.childElementCount > MAX_LOG_ROWS) {
            consoleRows.removeChild(consoleRows.firstElementChild);
        }
        if (followTail) {
            consoleRows.scrollTop = consoleRows.scrollHeight;
        }
    });

    q('.run-console-clear').addEventListener('click', () => {
        consoleRows.innerHTML = '';
        followTail = true;
    });

    // %% the prompt dialog

    // appended to the body so the dialog covers the page whichever tab is active
    const overlay = el('div', 'run-prompt-overlay');
    overlay.hidden = true;
    overlay.innerHTML = `
        <div class="run-prompt-dialog" role="dialog" aria-modal="true">
            <h2>operator prompt</h2>
            <pre class="run-prompt-text"></pre>
            <input type="text" class="text-input run-prompt-input" autocomplete="off">
            <div class="run-prompt-buttons">
                <button type="button" class="button run-prompt-yes" hidden>y</button>
                <button type="button" class="button run-prompt-no" hidden>n</button>
                <button type="button" class="button run-prompt-answer">Answer</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    const promptText = overlay.querySelector('.run-prompt-text');
    const promptInput = overlay.querySelector('.run-prompt-input');
    const yesButton = overlay.querySelector('.run-prompt-yes');
    const noButton = overlay.querySelector('.run-prompt-no');
    const answerButton = overlay.querySelector('.run-prompt-answer');

    let shownPromptId = null;

    function setPromptEnabled(enabled) {
        promptInput.disabled = !enabled;
        yesButton.disabled = !enabled;
        noButton.disabled = !enabled;
        answerButton.disabled = !enabled;
    }

    function renderPrompt(prompt) {
        if (!prompt) {
            overlay.hidden = true;
            shownPromptId = null;
            return;
        }
        if (prompt.id !== shownPromptId) {
            shownPromptId = prompt.id;
            promptText.textContent = prompt.text;
            promptInput.value = '';
            const yesNo = /\(y\/n\)/i.test(prompt.text);
            yesButton.hidden = !yesNo;
            noButton.hidden = !yesNo;
        }
        // the input is disabled from the answer until this next state message,
        // which follows the answer whether the server takes or refuses it
        setPromptEnabled(true);
        overlay.hidden = false;
        if (!promptInput.disabled) {
            promptInput.focus();
        }
    }

    function answer(text) {
        if (shownPromptId === null || promptInput.disabled) {
            return;
        }
        app.send({ type: 'prompt_answer', id: shownPromptId, text });
        setPromptEnabled(false);
    }

    answerButton.addEventListener('click', () => answer(promptInput.value));
    yesButton.addEventListener('click', () => answer('y'));
    noButton.addEventListener('click', () => answer('n'));
    promptInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            answer(promptInput.value);
        }
    });

    applyLevelFilter();
}
