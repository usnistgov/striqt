// JSON pointers, YAML text, and the edit messages of the Spec tab; nothing here
// touches the DOM.
//
// Every scalar is shown with String(value) and sent back as the text the user typed,
// because the YAML 1.1 loader leaves `3750e6` and `1/8000` as strings and a
// reformatted number would reach the file in another spelling. The edit builders
// take a `ctx` with `tree` (the merged spec) and `send(edits, reviewed)`.

import { isObject } from './util.js';

// kinds with typed text sent as written, without the quoting of a string scalar
const RAW_KINDS = new Set(['number', 'integer', 'fraction']);

// %% pointers and values

// RFC 6901 pointer of a key path
export function pointer(parts) {
    return parts
        .map((p) => '/' + String(p).replace(/~/g, '~0').replace(/\//g, '~1'))
        .join('');
}

export function child(base, key) {
    return base + pointer([key]);
}

export function parentPointer(ptr) {
    const i = ptr.lastIndexOf('/');
    return i <= 0 ? '' : ptr.slice(0, i);
}

export function unescapeToken(token) {
    return token.replace(/~1/g, '/').replace(/~0/g, '~');
}

// the node of `tree` at `ptr`, or undefined when a key or index is absent
export function valueAt(tree, ptr) {
    if (ptr === '') {
        return tree;
    }
    let node = tree;
    for (const token of ptr.split('/').slice(1)) {
        if (node === null || typeof node !== 'object') {
            return undefined;
        }
        const key = unescapeToken(token);
        node = Array.isArray(node) ? node[Number(key)] : node[key];
        if (node === undefined) {
            return undefined;
        }
    }
    return node;
}

export function shortName(owner) {
    return owner.split('.').pop();
}

// A whole-entry rewrite is built from the keys of the tree node only, so that a
// schema default never reaches the file as a key the user did not set: the keys of
// `item` that `fields` declare, with the values of the tree.
export function presentKeys(item, fields) {
    const kept = {};
    if (!isObject(item)) {
        return kept;
    }
    for (const field of fields) {
        if (item[field.name] !== undefined) {
            kept[field.name] = item[field.name];
        }
    }
    return kept;
}

// %% YAML text

// text that YAML 1.1 reads as a non-string plain scalar, or that contains a flow
// indicator
export function needsQuotes(text) {
    if (/^(["']).*\1$/s.test(text)) {
        return false;
    }
    return /^(?:~|null|true|false|yes|no|on|off|y|n)$/i.test(text)
        || /^[-+.\d]/.test(text)
        || /^[[\]{}#&*!|>'"%@`,?:]/.test(text)
        || /[,[\]{}]|: |\s#/.test(text)
        || text !== text.trim();
}

export function scalarText(value, kind) {
    if (value === null || value === undefined) {
        return 'null';
    }
    if (typeof value === 'string') {
        return !RAW_KINDS.has(kind) && needsQuotes(value) ? JSON.stringify(value) : value;
    }
    return String(value);
}

// YAML flow text of `value`; string elements of a list follow `kind`
export function flowText(value, kind) {
    if (Array.isArray(value)) {
        return '[' + value.map((v) => flowText(v, kind)).join(', ') + ']';
    }
    if (isObject(value)) {
        const entries = Object.entries(value).map(
            ([k, v]) => `${scalarText(k, 'string')}: ${flowText(v)}`);
        return '{' + entries.join(', ') + '}';
    }
    return scalarText(value, kind);
}

export function itemKind(field) {
    return field.item ? field.item.kind : undefined;
}

// YAML flow text of a whole struct entry, with the number-kind values of `fields`
// written as read from the file (`6.144e6` rather than `"6.144e6"`)
export function entryText(item, fields) {
    const kinds = new Map(fields.map((f) => [f.name, f.kind === 'list' ? itemKind(f) : f.kind]));
    const entries = Object.entries(item).map(
        ([k, v]) => `${scalarText(k, 'string')}: ${flowText(v, kinds.get(k))}`);
    return '{' + entries.join(', ') + '}';
}

// the text shown in the control of `field` for a value of the tree
export function displayText(value, field) {
    if (value === undefined) {
        return '';
    }
    if (Array.isArray(value) || isObject(value)) {
        return flowText(value, itemKind(field));
    }
    return String(value);
}

// the YAML text sent for the text typed into the control of `field`
export function typedText(field, text) {
    if (field.kind === 'list' || RAW_KINDS.has(field.kind)) {
        return text;
    }
    return needsQuotes(text) ? JSON.stringify(text) : text;
}

export function portCount(capture) {
    return capture && Array.isArray(capture.port) ? capture.port.length : 1;
}

// %% edits

// The server sets one key under an existing mapping. A key under a mapping that
// is absent from the tree is written as a flow mapping on the nearest present
// ancestor instead.
export function setEdit(ptr, valueText, ctx) {
    let target = ptr;
    let text = valueText;
    for (;;) {
        const parent = parentPointer(target);
        if (parent === '' || valueAt(ctx.tree, parent) !== undefined) {
            break;
        }
        const key = unescapeToken(target.slice(parent.length + 1));
        text = `{${scalarText(key, 'string')}: ${text}}`;
        target = parent;
    }
    return { pointer: target, op: 'set', value_text: text };
}

export function insertEdit(base, valueText, ctx) {
    if (Array.isArray(valueAt(ctx.tree, base))) {
        return { pointer: `${base}/-`, op: 'insert', value_text: valueText };
    }
    return setEdit(base, `[${valueText}]`, ctx);
}

// On an empty cell, the edit deletes the key when None or the default is the
// value anyway; otherwise the edit sets None, so that validation names the field.
export function commit(field, ptr, present, text, ctx) {
    const trimmed = text.trim();
    if (trimmed === '') {
        if (!present) {
            return;
        }
        if (field.default === null || field.optional) {
            ctx.send([{ pointer: ptr, op: 'delete' }]);
        } else {
            ctx.send([setEdit(ptr, 'null', ctx)]);
        }
        return;
    }
    ctx.send([setEdit(ptr, typedText(field, trimmed), ctx)]);
}

// One edit message sets the port count of a capture to `n`: `port` is renumbered
// 0..n-1 (the first port alone when n is 1), and each split per-port field of the
// capture is padded with its last value or truncated to n (its first value alone
// when n is 1) and marked for review.
export function resizePorts(capture, n, ctx) {
    const portPtr = child(capture.pointer, 'port');
    const port = capture.item ? capture.item.port : undefined;
    const first = Array.isArray(port) ? port[0] : (port === undefined || port === null ? 0 : port);
    const portText = n === 1
        ? scalarText(first, 'integer')
        : flowText(Array.from({ length: n }, (_, k) => k), 'integer');
    const edits = [setEdit(portPtr, portText, ctx)];
    const reviewed = [];
    for (const other of capture.fields) {
        if (!other.per_port || other.name === 'port') {
            continue;
        }
        const otherPtr = child(capture.pointer, other.name);
        const current = valueAt(ctx.tree, otherPtr);
        if (!Array.isArray(current) || current.length === 0) {
            continue;
        }
        let text;
        if (n === 1) {
            text = scalarText(current[0], other.kind);
        } else if (current.length === n) {
            continue;
        } else {
            const last = current[current.length - 1];
            const resized = current.length > n
                ? current.slice(0, n)
                : [...current, ...Array(n - current.length).fill(last)];
            text = flowText(resized, other.kind);
        }
        edits.push({ pointer: otherPtr, op: 'set', value_text: text });
        reviewed.push(otherPtr);
    }
    ctx.send(edits, reviewed);
}
