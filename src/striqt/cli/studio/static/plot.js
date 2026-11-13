// the plotly wrapper shared by the Preview and Run tabs: the dark layout, the WebGL
// fallback, and one draw function per container

import { el, isObject } from './util.js';

// the theme colors are the custom properties of `:root` in studio.css, read once
// when the module loads (the stylesheets are applied before a module script runs)
const theme = getComputedStyle(document.documentElement);
const color = (name) => theme.getPropertyValue(name).trim();

// Plotly layout defaults for the dark theme. Axis styling is in axisDefaults and is
// applied to every axis key (xaxis, yaxis, xaxis2, ...) by buildLayout().
const darkLayout = {
    paper_bgcolor: color('--bg-panel'),
    plot_bgcolor: color('--bg'),
    font: { color: color('--fg') },
    // the top margin holds the mode bar row with the variable dropdown
    margin: { t: 64, r: 50, b: 60, l: 70 }
};

const axisDefaults = { gridcolor: color('--plot-grid'), zerolinecolor: color('--plot-zeroline') };

// Line traces arrive as 'scattergl' (WebGL), which fails on a client without a WebGL
// context; those are downgraded to SVG 'scatter' in draw(). Heatmaps are SVG.
export const webglSupported = (() => {
    try {
        const canvas = document.createElement('canvas');
        return !!(window.WebGLRenderingContext &&
            (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) {
        return false;
    }
})();

// Recursive merge: recurse into plain objects; an array or primitive in the source
// replaces the target value. Arrays (annotations, shapes) are replaced wholesale
// because the backend regenerates them every frame.
export function deepMerge(target, source) {
    const out = { ...target };
    for (const key of Object.keys(source)) {
        const sv = source[key];
        const tv = out[key];
        out[key] = (isObject(sv) && isObject(tv)) ? deepMerge(tv, sv) : sv;
    }
    return out;
}

// Merge the dark theme with the incoming layout, then apply the grid styling to every
// axis key present. A backend-provided title/domain/range/type replaces the
// axisDefaults entry because it spreads last.
export function buildLayout(incoming) {
    const merged = deepMerge(darkLayout, incoming);
    for (const key of Object.keys(merged)) {
        if (/^[xy]axis\d*$/.test(key) && isObject(merged[key])) {
            merged[key] = { ...axisDefaults, ...merged[key] };
        }
    }
    return merged;
}

// A plot bound to one container element. `draw(plotData)` takes the `data` list of a
// `plot_data` message (figures from `Figure.to_json`) and draws the first figure;
// `reset()` makes the next draw a fresh plot; `resize()` follows the window.
export class Plot {
    constructor(container) {
        this.container = container;
        this.initialized = false;
    }

    reset() {
        this.initialized = false;
    }

    draw(plotData) {
        if (!plotData || plotData.length === 0) {
            return;
        }
        const figure = plotData[0];
        // fig.to_json() nests traces under 'data', not 'traces'.
        const traces = figure.data;
        // Bridge NaN gaps when falling back to SVG: the backend converts Inf -> NaN
        // (empty dB bins) and an SVG line breaks at every gap.
        if (!webglSupported && Array.isArray(traces)) {
            traces.forEach(t => {
                if (t.type === 'scattergl') {
                    t.type = 'scatter';
                    t.connectgaps = true;
                }
            });
        }
        const layout = buildLayout(figure.layout);

        if (!this.initialized) {
            Plotly.newPlot(this.container, traces, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            });
            this.initialized = true;
        } else {
            Plotly.react(this.container, traces, layout);
        }
    }

    showError(message, hint) {
        this.container.innerHTML = '';
        const box = el('div', 'loading');
        box.append(el('div', 'loading-error', `\u26A0\uFE0F ${message}`), el('div', 'loading-hint', hint || ''));
        this.container.append(box);
        this.initialized = false;
    }

    resize() {
        if (this.initialized) {
            Plotly.Plots.resize(this.container);
        }
    }
}
