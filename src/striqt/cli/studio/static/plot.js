// the plotly wrapper shared by the Preview and Run tabs: the dark layout, the WebGL
// fallback, and one draw function per container

// Plotly layout defaults for the dark theme. Axis styling is in axisDefaults and is
// applied to every axis key (xaxis, yaxis, xaxis2, ...) by buildLayout().
const darkLayout = {
    paper_bgcolor: '#16213e',
    plot_bgcolor: '#1a1a2e',
    font: { color: '#eee' },
    // the top margin holds the mode bar row with the variable dropdown
    margin: { t: 64, r: 50, b: 60, l: 70 }
};

const axisDefaults = { gridcolor: '#2a2a4e', zerolinecolor: '#3a3a5e' };

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

function isPlainObject(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
}

// Recursive merge: recurse into plain objects; an array or primitive in the source
// replaces the target value. Arrays (annotations, shapes) are replaced wholesale
// because the backend regenerates them every frame.
export function deepMerge(target, source) {
    const out = { ...target };
    for (const key of Object.keys(source)) {
        const sv = source[key];
        const tv = out[key];
        out[key] = (isPlainObject(sv) && isPlainObject(tv)) ? deepMerge(tv, sv) : sv;
    }
    return out;
}

// Merge the dark theme with the incoming layout, then apply the grid styling to every
// axis key present. A backend-provided title/domain/range/type replaces the
// axisDefaults entry because it spreads last.
export function buildLayout(incoming) {
    const merged = deepMerge(darkLayout, incoming);
    for (const key of Object.keys(merged)) {
        if (/^[xy]axis\d*$/.test(key) && isPlainObject(merged[key])) {
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
        this.container.innerHTML = `
            <div class="loading" style="flex-direction: column; gap: 1rem;">
                <div style="color: #e94560; font-size: 1.2rem;">&#x26A0;&#xFE0F; ${message}</div>
                <div style="color: #666; font-size: 0.9rem;">${hint || ''}</div>
            </div>
        `;
        this.initialized = false;
    }

    resize() {
        if (this.initialized) {
            Plotly.Plots.resize(this.container);
        }
    }
}
