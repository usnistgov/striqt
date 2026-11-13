// the Preview tab: the live plot of the selected variable, the loop and capture
// selectors, and the table of the capture fields that differ between the entries

import { Plot } from './plot.js';
import { el, busy } from './util.js';

export function init(app) {
    const plotContainer = document.getElementById('plot-container');
    const variableSelect = document.getElementById('variable-select');
    const pauseButton = document.getElementById('pause-button');
    const coordsSidebar = document.getElementById('coords-sidebar');
    const captureControls = document.getElementById('capture-controls');

    const plot = new Plot(plotContainer);
    let running = true;
    let currentVariable = '';

    app.on('connection', (state) => {
        pauseButton.disabled = state !== 'connected';
    });

    app.on('state', (msg) => {
        updateVariableSelect(msg.available_variables, msg.variable_labels, msg.selected_variable);
        updateCaptureSelects(msg.capture_options, msg.capture_selection);
        updateRunButton(msg.running !== false, msg.mode);
        updateCoordsTable(msg.coordinates);
    });

    app.on('plot_data', (msg) => {
        plot.draw(msg.data);
        if (msg.coordinates) {
            updateCoordsTable(msg.coordinates);
        }
    });

    // an error that answers no command is a plot failure; a refused command is shown
    // in the error bar of the page shell alone
    app.on('error', (msg) => {
        if (!msg.command) {
            plot.showError(msg.message, 'try selecting a different data variable');
        }
    });

    app.onTab('preview', () => plot.resize());
    window.addEventListener('resize', () => plot.resize());

    function updateRunButton(isRunning, mode) {
        running = isRunning;
        pauseButton.textContent = isRunning ? 'Pause' : 'Resume';
        pauseButton.classList.toggle('running', isRunning);
        pauseButton.disabled = !app.connected() || busy(mode);
    }

    pauseButton.addEventListener('click', () => {
        app.send({ type: 'set_running', running: !running });
    });

    // The capture fields that differ between the captures: entries, one column per
    // port. `coordinatesData` is null until the first acquisition is analyzed.
    function updateCoordsTable(coordinatesData) {
        if (!coordinatesData || !coordinatesData.fields || coordinatesData.fields.length === 0) {
            const text = coordinatesData
                ? 'No capture field differs between the entries'
                : 'No capture data yet';
            coordsSidebar.innerHTML = `<div class="coords-section"><p class="placeholder">${text}</p></div>`;
            return;
        }
        const fields = coordinatesData.fields;

        const numPorts = coordinatesData.num_ports || 1;
        let html = `
            <div class="coords-section">
                <table class="coords-table">
                    <thead>
                        <tr>
                            <th class="coord-name"></th>
        `;
        for (let i = 0; i < numPorts; i++) {
            html += `<th class="coord-value">Port ${i}</th>`;
        }
        html += `<th class="coord-unit"></th></tr></thead><tbody>`;
        for (const coord of fields) {
            html += `<tr><td class="coord-name">${coord.name}</td>`;
            for (let i = 0; i < numPorts; i++) {
                html += `<td class="coord-value">${coord.values[i]}</td>`;
            }
            html += `<td class="coord-unit">${coord.unit || ''}</td></tr>`;
        }
        html += `</tbody></table></div>`;
        coordsSidebar.innerHTML = html;
    }

    // One <select> per loop field and a slider for the captures: entry (when the
    // sweep lists more than one). Values are indexes into the server's option lists.
    function buildSelect(id, labelText, labels, selectedIndex) {
        const group = el('div', 'control-group');
        const label = el('label', '', labelText);
        label.htmlFor = id;
        const select = el('select');
        select.id = id;
        labels.forEach((text, i) => {
            const option = el('option', '', text);
            option.value = String(i);
            select.append(option);
        });
        select.value = String(selectedIndex || 0);
        select.addEventListener('change', sendCaptureSelection);
        group.append(label, select);
        return group;
    }

    function buildCaptureSlider(count, selectedIndex) {
        const group = el('div', 'control-group');
        const label = el('label', '', 'capture');
        label.htmlFor = 'capture-select';
        const input = el('input');
        input.type = 'range';
        input.id = 'capture-select';
        input.min = '0';
        input.max = String(count - 1);
        input.step = '1';
        input.value = String(selectedIndex || 0);
        const value = el('span', 'slider-value');
        const show = () => { value.textContent = `${input.value} / ${count - 1}`; };
        show();
        // the value follows the thumb; the selection is sent on release
        input.addEventListener('input', show);
        input.addEventListener('change', sendCaptureSelection);
        const slider = el('div', 'slider');
        slider.append(input, value);
        group.append(label, slider);
        return group;
    }

    function updateCaptureSelects(options, selection) {
        captureControls.innerHTML = '';
        if (!options) {
            return;
        }
        const sel = selection || { capture: 0, loops: [] };
        // the loops nest outside the captures: entry, so the slider follows them
        (options.loops || []).forEach((loop, j) => {
            if (loop.labels.length === 0) {
                return;
            }
            const chosen = (sel.loops && sel.loops.length > j) ? sel.loops[j] : 0;
            captureControls.appendChild(
                buildSelect(`loop-select-${j}`, `${loop.field}:`, loop.labels, chosen));
        });
        if (options.captures && options.captures.length > 1) {
            captureControls.appendChild(buildCaptureSlider(options.captures.length, sel.capture));
        }
    }

    function indexOf(element) {
        return element ? parseInt(element.value, 10) : null;
    }

    function sendCaptureSelection() {
        const loops = [];
        for (let j = 0; ; j++) {
            const select = document.getElementById(`loop-select-${j}`);
            if (!select) {
                break;
            }
            loops.push(indexOf(select));
        }
        app.send({
            type: 'select_captures',
            capture: indexOf(document.getElementById('capture-select')),
            loops: loops
        });
    }

    function updateVariableSelect(variables, labels, selected) {
        variableSelect.innerHTML = '';
        if (!variables || variables.length === 0) {
            const option = el('option', '', 'Waiting for data...');
            option.value = '';
            variableSelect.append(option);
            return;
        }
        variables.forEach(v => {
            const option = el('option', '', (labels && labels[v]) ? labels[v] : v.replace(/_/g, ' '));
            option.value = v;
            if (v === selected) {
                option.selected = true;
                currentVariable = v;
            }
            variableSelect.append(option);
        });
    }

    variableSelect.addEventListener('change', (e) => {
        const newVariable = e.target.value;
        if (newVariable && newVariable !== currentVariable) {
            currentVariable = newVariable;
            plot.reset();
            app.send({ type: 'select_variable', variable: newVariable });
        }
    });
}
