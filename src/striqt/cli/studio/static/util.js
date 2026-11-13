// DOM and value helpers shared by the tab modules

export function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text !== undefined) {
        node.textContent = text;
    }
    return node;
}

export function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// append `text` to the tooltip of `node` on its own line
export function addTitle(node, text) {
    if (text) {
        node.title = node.title ? `${node.title}\n${text}` : text;
    }
}

export function iconButton(glyph, title, onClick, className = '') {
    const button = el('button', `icon-button ${className}`.trim(), glyph);
    button.type = 'button';
    button.title = title;
    button.addEventListener('click', onClick);
    return button;
}

// a read-only JSON rendering of `value`
export function preOf(value) {
    return el('pre', 'opaque', value === undefined ? '' : JSON.stringify(value, null, 1));
}

// whether the session `mode` of a state message blocks edits, applies, and starts
export function busy(mode) {
    return mode === 'running' || mode === 'prompting';
}
