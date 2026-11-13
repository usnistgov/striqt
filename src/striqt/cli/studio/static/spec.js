// the Spec tab: a grid editor over the editor schema and the merged spec tree.
//
// Every scalar is shown with String(value) and sent back as the text the user typed,
// because the YAML 1.1 loader leaves `3750e6` and `1/8000` as strings and a
// reformatted number would reach the file in another spelling.

// kinds with typed text sent as written, without the quoting of a string scalar
const RAW_KINDS = new Set(['number', 'integer', 'fraction']);

const READ_ONLY_GLYPH = '\u{1F512}';

// %% pointers and values

// RFC 6901 pointer of a key path
export function pointer(parts) {
    return parts
        .map((p) => '/' + String(p).replace(/~/g, '~0').replace(/\//g, '~1'))
        .join('');
}

function child(base, key) {
    return base + pointer([key]);
}

function parentPointer(ptr) {
    const i = ptr.lastIndexOf('/');
    return i <= 0 ? '' : ptr.slice(0, i);
}

function unescapeToken(token) {
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

function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

// A whole-entry rewrite is built from the keys of the tree node only, so that a
// schema default never reaches the file as a key the user did not set: the keys of
// `item` that `fields` declare, with the values of the tree.
function presentKeys(item, fields) {
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
function needsQuotes(text) {
    if (/^(["']).*\1$/s.test(text)) {
        return false;
    }
    return /^(?:~|null|true|false|yes|no|on|off|y|n)$/i.test(text)
        || /^[-+.\d]/.test(text)
        || /^[[\]{}#&*!|>'"%@`,?:]/.test(text)
        || /[,[\]{}]|: |\s#/.test(text)
        || text !== text.trim();
}

function scalarText(value, kind) {
    if (value === null || value === undefined) {
        return 'null';
    }
    if (typeof value === 'string') {
        return !RAW_KINDS.has(kind) && needsQuotes(value) ? JSON.stringify(value) : value;
    }
    return String(value);
}

// YAML flow text of `value`; string elements of a list follow `kind`
function flowText(value, kind) {
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

function itemKind(field) {
    return field.item ? field.item.kind : undefined;
}

// YAML flow text of a whole struct entry, with the number-kind values of `fields`
// written as read from the file (`6.144e6` rather than `"6.144e6"`)
function entryText(item, fields) {
    const kinds = new Map(fields.map((f) => [f.name, f.kind === 'list' ? itemKind(f) : f.kind]));
    const entries = Object.entries(item).map(
        ([k, v]) => `${scalarText(k, 'string')}: ${flowText(v, kinds.get(k))}`);
    return '{' + entries.join(', ') + '}';
}

// the text shown in the control of `field` for a value of the tree
function displayText(value, field) {
    if (value === undefined) {
        return '';
    }
    if (Array.isArray(value) || isObject(value)) {
        return flowText(value, itemKind(field));
    }
    return String(value);
}

// the YAML text sent for the text typed into the control of `field`
function typedText(field, text) {
    if (field.kind === 'list' || RAW_KINDS.has(field.kind)) {
        return text;
    }
    return needsQuotes(text) ? JSON.stringify(text) : text;
}

function portCount(capture) {
    return capture && Array.isArray(capture.port) ? capture.port.length : 1;
}

// %% DOM helpers

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text !== undefined) {
        node.textContent = text;
    }
    return node;
}

function addTitle(node, text) {
    if (text) {
        node.title = node.title ? `${node.title}\n${text}` : text;
    }
}

function iconButton(glyph, title, onClick, className = '') {
    const button = el('button', `icon-button ${className}`.trim(), glyph);
    button.type = 'button';
    button.title = title;
    button.addEventListener('click', onClick);
    return button;
}

function preOf(value) {
    return el('pre', 'opaque', value === undefined ? '' : JSON.stringify(value, null, 1));
}

function fieldTip(field) {
    return field.description || field.standard_name || '';
}

function isReadOnly(field) {
    return field.read_only
        || field.kind === 'opaque'
        || field.kind === 'dict'
        || (field.kind === 'struct' && (field.variants.length > 0 || field.fields.length === 0));
}

// the rows of a grid: one per field, with the rows of a nested struct after its own
// row; `owner` is the owner of the top-level field for the group label
function fieldRows(fields, path = [], depth = 0, owner = null) {
    const rows = [];
    for (const field of fields) {
        const groupOwner = depth === 0 ? field.owner : owner;
        const nested = field.kind === 'struct' && !isReadOnly(field);
        const fieldPath = [...path, field.name];
        rows.push({ field, path: fieldPath, depth, container: nested, owner: groupOwner });
        if (nested) {
            rows.push(...fieldRows(field.fields, fieldPath, depth + 1, groupOwner));
        }
    }
    return rows;
}

// %% the tab

export function init(app) {
    const panel = document.getElementById('tab-spec');
    panel.innerHTML = '';
    const nav = el('nav', 'spec-nav');
    const editor = el('div', 'spec-editor');
    const toolbar = el('div', 'spec-toolbar');
    const legend = el('div', 'spec-legend');
    const actions = el('div', 'spec-actions');
    const messages = el('div', 'spec-messages');
    const body = el('fieldset', 'spec-body');
    toolbar.append(legend, actions);
    editor.append(toolbar, messages, body);
    panel.append(nav, editor);

    let selected = null;
    let dirty = true;
    let lastError = null;
    let lastValidation = null;
    // pointers of per-port fields resized to a new port count, until edited
    const review = new Set();

    app.on('schema', scheduleRender);
    app.on('spec', () => {
        lastError = null;
        scheduleRender();
    });
    app.on('state', (msg) => {
        renderActions();
        const key = JSON.stringify([
            msg.validation, msg.editable, msg.changed_sections || [], msg.mode,
        ]);
        if (key !== lastValidation) {
            lastValidation = key;
            scheduleRender();
        } else {
            renderMessages(context());
        }
    });
    app.on('connection', renderActions);
    app.on('error', (msg) => {
        if (app.currentTab() === 'spec') {
            lastError = msg.message;
            renderMessages(context());
        }
    });
    app.onTab('spec', () => {
        if (dirty) {
            render();
        }
    });

    function scheduleRender() {
        if (panel.hidden) {
            dirty = true;
        } else {
            render();
        }
    }

    function sendEdits(edits, reviewed = []) {
        for (const edit of edits) {
            review.delete(edit.pointer);
            review.delete(parentPointer(edit.pointer));
        }
        for (const ptr of reviewed) {
            review.add(ptr);
        }
        app.send({ type: 'edit', edits });
    }

    function context() {
        const spec = app.latest.spec;
        const state = app.latest.state;
        const hasTree = !!(spec && spec.tree);
        return {
            tree: hasTree ? spec.tree : {},
            hasTree,
            origin: spec ? spec.origin : {},
            regions: spec ? spec.regions : {},
            files: spec ? spec.files : [],
            errors: state ? state.validation.errors : [],
            editable: !!(state && state.editable) && hasTree,
            // the sections with file values that differ from the spec applied last
            changed: new Set(state ? state.changed_sections || [] : []),
            busy: !!state && (state.mode === 'running' || state.mode === 'prompting'),
            review,
            send: sendEdits,
        };
    }

    // %% edits

    // The server sets one key under an existing mapping. A key under a mapping that
    // is absent from the tree is written as a flow mapping on the nearest present
    // ancestor instead.
    function setEdit(ptr, valueText, ctx) {
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

    function insertEdit(base, valueText, ctx) {
        if (Array.isArray(valueAt(ctx.tree, base))) {
            return { pointer: `${base}/-`, op: 'insert', value_text: valueText };
        }
        return setEdit(base, `[${valueText}]`, ctx);
    }

    // On an empty cell, the edit deletes the key when None or the default is the
    // value anyway; otherwise the edit sets None, so that validation names the field.
    function commit(field, ptr, present, text, ctx) {
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
    function resizePorts(capture, n, ctx) {
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

    // %% provenance and validation marks

    // the file index that holds `ptr`, or that would receive a new key at `ptr`: the
    // last file of the nearest region, else the origin of the nearest ancestor
    function fileOf(ptr, ctx) {
        for (let p = ptr; ; p = parentPointer(p)) {
            const region = ctx.regions[p];
            if (region && region.length) {
                return region[region.length - 1];
            }
            const origin = ctx.origin[p];
            if (origin) {
                return origin[0];
            }
            if (p === '') {
                return null;
            }
        }
    }

    function markProvenance(node, ptr, ctx) {
        const index = fileOf(ptr, ctx);
        if (index === null) {
            return;
        }
        node.dataset.file = String(index);
        const region = ctx.regions[ptr];
        if (region && region.length > 1) {
            node.classList.add('region');
            addTitle(node, 'merged from:\n' + region.map((i) => ctx.files[i]).join('\n'));
        } else {
            addTitle(node, ctx.files[index]);
        }
    }

    // the messages of the errors at `ptr` or under it; at `ptr` only when `exact`
    function errorsAt(ptr, ctx, exact = false) {
        const found = [];
        for (const error of ctx.errors) {
            const where = [error.path, ...(error.locations || [])];
            if (where.some((w) => w && (w === ptr || (!exact && w.startsWith(`${ptr}/`))))) {
                found.push(error.message);
            }
        }
        return found;
    }

    function markErrors(node, ptr, ctx, exact = false) {
        const found = errorsAt(ptr, ctx, exact);
        if (found.length) {
            node.classList.add('invalid');
            addTitle(node, found.join('\n'));
        }
    }

    // %% rendering

    function render() {
        dirty = false;
        const schema = app.latest.schema ? app.latest.schema.schema : null;
        const ctx = context();
        renderNav(schema, ctx);
        renderLegend(ctx);
        renderActions();
        renderMessages(ctx);

        const focus = focusedControl();
        const scroll = [body.scrollTop, body.scrollLeft];
        body.innerHTML = '';
        body.disabled = !ctx.editable;
        if (!schema) {
            body.append(el('p', 'placeholder', 'Waiting for the schema...'));
            return;
        }
        if (!app.latest.spec) {
            body.append(el('p', 'placeholder', 'Waiting for the spec...'));
            return;
        }
        if (!ctx.hasTree) {
            body.append(el('p', 'note', 'JSON specs are read-only in the page'));
        }
        const section = schema.sections.find((s) => s.name === selected);
        if (section) {
            body.append(renderSection(section, ctx));
        }
        [body.scrollTop, body.scrollLeft] = scroll;
        restoreFocus(focus);
    }

    // the control under edit when a message re-renders the grid, with text the user
    // typed since the last commit
    function focusedControl() {
        const active = document.activeElement;
        if (!active || !body.contains(active) || !active.dataset.pointer) {
            return null;
        }
        const typed = active.dataset.initial !== undefined && active.value !== active.dataset.initial;
        return {
            pointer: active.dataset.pointer,
            value: typed ? active.value : null,
            start: active.selectionStart,
        };
    }

    function restoreFocus(focus) {
        if (!focus) {
            return;
        }
        const selector = `input[data-pointer="${CSS.escape(focus.pointer)}"], `
            + `select[data-pointer="${CSS.escape(focus.pointer)}"]`;
        const again = body.querySelector(selector);
        if (!again) {
            return;
        }
        if (focus.value !== null) {
            again.value = focus.value;
        }
        again.focus();
        if (typeof focus.start === 'number' && again.type === 'text') {
            again.setSelectionRange(focus.start, focus.start);
        }
    }

    function renderNav(schema, ctx) {
        nav.innerHTML = '';
        if (!schema) {
            return;
        }
        if (!schema.sections.some((s) => s.name === selected)) {
            const first = schema.sections.find((s) => s.name === 'captures')
                || schema.sections.find((s) => !s.read_only)
                || schema.sections[0];
            selected = first ? first.name : null;
        }
        const title = el('div', 'spec-nav-title', shortName(schema.action_cls));
        title.title = schema.action_cls;
        nav.append(title);
        for (const section of schema.sections) {
            const button = el('button', 'spec-nav-item');
            button.type = 'button';
            button.append(el('span', '', section.name));
            if (section.read_only) {
                button.classList.add('read-only');
                button.append(el('span', 'lock-glyph', READ_ONLY_GLYPH));
            }
            if (section.name === selected) {
                button.classList.add('active');
            }
            if (ctx.changed.has(section.name)) {
                button.classList.add('changed');
                button.title = 'edited since the last apply';
            }
            if (errorsAt(pointer([section.name]), ctx).length) {
                button.classList.add('invalid');
            }
            button.addEventListener('click', () => {
                selected = section.name;
                render();
            });
            nav.append(button);
        }
    }

    function renderLegend(ctx) {
        legend.innerHTML = '';
        ctx.files.forEach((name, i) => {
            const item = el('span', 'legend-item');
            const swatch = el('span', 'swatch');
            swatch.dataset.file = String(i);
            item.append(swatch, el('span', 'legend-name', name));
            item.title = name;
            legend.append(item);
        });
    }

    function renderActions() {
        actions.innerHTML = '';
        const state = app.latest.state;
        if (!state) {
            return;
        }
        if (!state.editable) {
            actions.append(el('span', 'note', 'JSON specs are read-only in the page'));
        }
        if (state.applied_differs) {
            actions.append(el('span', 'badge differs', 'edited spec differs from the applied one'));
        }
        const apply = el('button', 'apply-button', 'Apply');
        apply.type = 'button';
        const busy = state.mode === 'running' || state.mode === 'prompting';
        apply.disabled = !(state.validation.ok && state.editable && app.connected() && !busy);
        apply.title = busy ? 'stop the run before applying the spec'
            : !state.validation.ok ? 'the spec files do not validate'
                : 'apply the edited spec to the acquisition';
        apply.addEventListener('click', () => app.send({ type: 'apply' }));
        actions.append(apply);
    }

    function renderMessages(ctx) {
        messages.innerHTML = '';
        const texts = ctx.errors.map((e) => (e.path ? `${e.path}: ` : '') + e.message);
        if (lastError) {
            texts.unshift(lastError);
        }
        messages.hidden = texts.length === 0;
        for (const text of texts) {
            messages.append(el('div', 'message', text));
        }
    }

    function renderSection(section, ctx) {
        switch (section.kind) {
            case 'struct_list':
                return renderStructList(section, ctx);
            case 'tagged_list':
                return renderTaggedList(section, ctx);
            case 'struct':
                return renderStruct(section, ctx);
            case 'optional_struct_dict':
                return renderOptionalStructDict(section, ctx);
            case 'dict':
                return renderDict(section, ctx);
            default:
                return renderOpaque(section, ctx);
        }
    }

    function sectionShell(section, ctx) {
        const wrap = el('section', 'spec-section');
        const header = el('header', 'section-header');
        const ptr = pointer([section.name]);
        header.append(el('h2', '', section.name));
        if (ctx.changed.has(section.name) && ctx.editable) {
            const revert = el('button', 'revert-button', 'Revert');
            revert.type = 'button';
            revert.title = 'discard the edits to this section since the last apply';
            revert.disabled = ctx.busy;
            revert.addEventListener('click', () => {
                app.send({ type: 'revert', section: section.name });
            });
            header.append(revert);
        }
        if (section.read_only) {
            header.append(el('span', 'lock-glyph', READ_ONLY_GLYPH), el('span', 'muted', 'read-only'));
        }
        addTitle(header, fieldTip(section.field));
        markProvenance(header, ptr, ctx);
        markErrors(header, ptr, ctx);
        wrap.append(header);
        return wrap;
    }

    // One table: the rows of `fields` grouped by owner with a vertical class label,
    // and one value column per entry of `columns` ({pointer, label, span, capture,
    // removable}); a column with `span` > 1 has one sub-column per port, which the
    // split per-port fields of a capture occupy one cell each. `options.heads` adds
    // the column heading row, `options.add` a trailing column with an add button,
    // and `options.single` the class of a one-column grid.
    function renderGrid(fields, columns, ctx, options = {}) {
        const table = el('table', 'spec-grid');
        if (options.single) {
            table.classList.add('single');
        }
        const rows = fieldRows(fields);
        if (options.heads) {
            const thead = el('thead');
            const tr = el('tr');
            tr.append(el('th', 'group-label'), el('th', 'field-name'));
            for (const column of columns) {
                const th = el('th', 'column-head');
                th.colSpan = column.span || 1;
                th.append(el('span', 'column-label', column.label));
                if (column.capture) {
                    th.append(portStepper(column, ctx));
                }
                markProvenance(th, column.pointer, ctx);
                markErrors(th, column.pointer, ctx);
                if (column.removable) {
                    th.append(iconButton('×', `remove ${column.label}`, () => {
                        ctx.send([{ pointer: column.pointer, op: 'delete' }]);
                    }, 'remove'));
                }
                tr.append(th);
            }
            if (options.add) {
                const th = el('th', 'column-add');
                th.append(iconButton('+', options.addTitle || 'add an entry', options.add, 'add'));
                tr.append(th);
            }
            thead.append(tr);
            table.append(thead);
        }
        const tbody = el('tbody');
        rows.forEach((row, index) => {
            const tr = el('tr');
            if (index === 0 || rows[index - 1].owner !== row.owner) {
                let span = 1;
                while (index + span < rows.length && rows[index + span].owner === row.owner) {
                    span += 1;
                }
                const th = el('th', 'group-label');
                th.rowSpan = span;
                th.append(el('span', 'vertical', shortName(row.owner)));
                th.title = row.owner;
                tr.append(th);
            }
            const name = el('th', 'field-name');
            name.style.paddingLeft = `${0.5 + row.depth}rem`;
            name.append(document.createTextNode(row.field.name));
            if (row.field.units) {
                name.append(el('span', 'units', row.field.units));
            }
            if (row.field.required) {
                name.classList.add('required');
            }
            addTitle(name, fieldTip(row.field));
            if (row.field.per_port && row.field.name !== 'port' && columns.some((c) => c.capture)) {
                name.append(rowToggle(row.field, row, columns, ctx));
            }
            tr.append(name);
            for (const column of columns) {
                const ptr = column.pointer + pointer(row.path);
                const span = column.span || 1;
                if (row.container) {
                    const td = el('td', 'container');
                    td.colSpan = span;
                    markProvenance(td, ptr, ctx);
                    markErrors(td, ptr, ctx);
                    tr.append(td);
                } else if (row.field.per_port && column.capture) {
                    tr.append(...perPortCells(row.field, ptr, ctx, column));
                } else {
                    const td = valueCell(row.field, ptr, ctx);
                    td.colSpan = span;
                    tr.append(td);
                }
            }
            if (options.add) {
                tr.append(el('td', 'column-add'));
            }
            tbody.append(tr);
        });
        table.append(tbody);
        return table;
    }

    // the classes, color, and tooltips of a value cell at `ptr`
    function decorateCell(td, ptr, ctx) {
        const value = valueAt(ctx.tree, ptr);
        const present = value !== undefined;
        if (!present) {
            td.classList.add('default');
        }
        if (ctx.review.has(ptr)) {
            td.classList.add('review');
            addTitle(td, 'resized to the port count; check the values');
        }
        markProvenance(td, ptr, ctx);
        markErrors(td, ptr, ctx);
        return { value, present };
    }

    function valueCell(field, ptr, ctx) {
        const td = el('td', 'value');
        const { value, present } = decorateCell(td, ptr, ctx);
        if (isReadOnly(field)) {
            td.classList.add('read-only');
            td.append(preOf(present ? value : field.default));
        } else {
            td.append(control(field, ptr, value, present, ctx));
        }
        return td;
    }

    function textInput(ptr, text) {
        const input = el('input');
        input.type = 'text';
        input.spellcheck = false;
        input.dataset.pointer = ptr;
        input.value = text;
        input.dataset.initial = text;
        input.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                input.blur();
            }
        });
        return input;
    }

    // the input or select of a scalar or list field, sending on change
    function control(field, ptr, value, present, ctx) {
        const shown = present ? value : field.default;
        if (field.kind === 'literal' || field.kind === 'boolean') {
            const choices = field.kind === 'boolean' ? [true, false] : field.choices;
            const select = el('select');
            select.dataset.pointer = ptr;
            const index = choices.findIndex((c) => c === shown);
            if (field.optional || index < 0) {
                select.append(el('option', '', ''));
            }
            choices.forEach((choice, i) => {
                const option = el('option', '', String(choice));
                option.value = String(i);
                select.append(option);
            });
            select.value = index < 0 ? '' : String(index);
            select.addEventListener('change', () => {
                if (select.value === '') {
                    commit(field, ptr, present, '', ctx);
                } else {
                    const text = scalarText(choices[Number(select.value)], field.kind);
                    ctx.send([setEdit(ptr, text, ctx)]);
                }
            });
            return select;
        }
        const text = present ? displayText(value, field)
            : (field.default === null ? '' : displayText(field.default, field));
        const input = textInput(ptr, text);
        if (field.kind === 'list') {
            input.placeholder = '[a, b]';
        }
        input.addEventListener('change', () => commit(field, ptr, present, input.value, ctx));
        return input;
    }

    // the `ports N` stepper of a capture column; the port count is at least 1
    function portStepper(column, ctx) {
        const ports = column.span || 1;
        const box = el('span', 'stepper');
        const minus = iconButton('−', 'remove the last port', () => {
            resizePorts(column.capture, ports - 1, ctx);
        }, 'step');
        minus.disabled = ports <= 1;
        const plus = iconButton('+', 'add a port', () => {
            resizePorts(column.capture, ports + 1, ctx);
        }, 'step');
        box.append(minus, el('span', 'stepper-label', `ports ${ports}`), plus);
        return box;
    }

    // The toggle of a per-port row acts on every capture column at once. A click
    // splits each tied capture (N copies of its value or the schema default, N its
    // port count) while any capture is tied; otherwise it ties every capture to its
    // first value. The row is `split` when every capture holds a list, `mixed` when
    // some do.
    function rowToggle(field, row, columns, ctx) {
        const targets = columns.filter((c) => c.capture).map((c) => {
            const ptr = c.pointer + pointer(row.path);
            return { ptr, value: valueAt(ctx.tree, ptr), ports: c.span || 1 };
        });
        const lists = targets.filter((t) => Array.isArray(t.value)).length;
        const state = lists === 0 ? 'tied' : lists === targets.length ? 'split' : 'mixed';
        const button = el('button', 'link-toggle', '\u{1F517}');
        button.type = 'button';
        button.classList.toggle('split', state === 'split');
        button.classList.toggle('mixed', state === 'mixed');
        button.title = state === 'tied'
            ? 'one value for every port; click for one value per port'
            : state === 'split'
                ? 'one value per port; click for one value for every port'
                : 'mixed: click for one value per port';
        button.addEventListener('click', () => {
            const edits = [];
            for (const t of targets) {
                if (state === 'split') {
                    edits.push(setEdit(t.ptr, scalarText(t.value[0], field.kind), ctx));
                } else if (!Array.isArray(t.value)) {
                    const one = t.value === undefined ? field.default : t.value;
                    edits.push(setEdit(t.ptr, flowText(Array(t.ports).fill(one), field.kind), ctx));
                }
            }
            if (edits.length) {
                ctx.send(edits);
            }
        });
        return button;
    }

    // The cells of a per-port field in the sub-columns of a capture: one spanning
    // cell for a tied value (a scalar or an absent key), one cell per port for a split
    // value (a list). The `port` field is split by its own length.
    function perPortCells(field, ptr, ctx, column) {
        const ports = column.span || 1;
        const value = valueAt(ctx.tree, ptr);
        if (!Array.isArray(value)) {
            const td = valueCell(field, ptr, ctx);
            td.colSpan = ports;
            return [td];
        }
        const mismatch = value.length !== ports;
        const cells = [];
        for (let k = 0; k < ports; k += 1) {
            const elementPtr = child(ptr, k);
            const td = el('td', k === 0 ? 'value sub' : 'value sub inner');
            if (k === 0) {
                markProvenance(td, ptr, ctx);
            }
            markErrors(td, elementPtr, ctx);
            markErrors(td, ptr, ctx, true);
            if (mismatch) {
                td.classList.add('review');
                addTitle(td, `${value.length} values for ${ports} ports`);
            } else if (ctx.review.has(ptr)) {
                td.classList.add('review');
                addTitle(td, 'resized to the port count; check the values');
            }
            const input = textInput(elementPtr, k < value.length ? String(value[k]) : '');
            input.addEventListener('change', () => {
                const t = input.value.trim();
                const text = t === '' ? 'null' : typedText(field, t);
                if (k < value.length) {
                    ctx.send([{ pointer: elementPtr, op: 'set', value_text: text }]);
                } else {
                    // the server sets an existing element only, so a value past the end
                    // of the list rewrites the list padded with null up to that index
                    const padded = [...value, ...Array(k + 1 - value.length).fill(null)];
                    const items = padded.map((v, j) => (j === k ? text : scalarText(v, field.kind)));
                    ctx.send([setEdit(ptr, `[${items.join(', ')}]`, ctx)]);
                }
            });
            td.append(input);
            cells.push(td);
        }
        return cells;
    }

    function renderStructList(section, ctx) {
        const base = pointer([section.name]);
        const items = valueAt(ctx.tree, base);
        const list = Array.isArray(items) ? items : [];
        const fields = section.schema.fields;
        const wrap = sectionShell(section, ctx);
        const columns = list.map((item, i) => ({
            pointer: child(base, i),
            label: `#${i}`,
            span: portCount(item),
            capture: { pointer: child(base, i), item, fields },
            removable: true,
        }));
        const add = () => {
            // the copy goes into the file as one flow mapping of the keys present in
            // the tree, so no schema default is written as a key
            const last = list[list.length - 1];
            const text = isObject(last) ? entryText(last, fields) : '{}';
            ctx.send([insertEdit(base, text, ctx)]);
        };
        wrap.append(renderGrid(fields, columns, ctx, {
            heads: true, add, addTitle: 'add a copy of the last entry',
        }));
        return wrap;
    }

    function renderStruct(section, ctx) {
        const wrap = sectionShell(section, ctx);
        if (section.schema.fields.length === 0) {
            wrap.append(el('p', 'placeholder', 'no fields'));
        } else {
            const columns = [{ pointer: pointer([section.name]) }];
            wrap.append(renderGrid(section.schema.fields, columns, ctx, { single: true }));
        }
        return wrap;
    }

    function entryHeader(ptr, label, ctx, onRemove) {
        const head = el('div', 'entry-header');
        head.append(el('span', 'entry-label', label));
        markProvenance(head, ptr, ctx);
        markErrors(head, ptr, ctx);
        if (onRemove) {
            head.append(iconButton('×', `remove ${label}`, onRemove, 'remove'));
        }
        return head;
    }

    function renderTaggedList(section, ctx) {
        const base = pointer([section.name]);
        const items = valueAt(ctx.tree, base);
        const list = Array.isArray(items) ? items : [];
        const schema = section.schema;
        const tag = schema.tag;
        const wrap = sectionShell(section, ctx);
        list.forEach((item, i) => {
            const ptr = child(base, i);
            const card = el('div', 'entry');
            const head = entryHeader(ptr, `#${i}`, ctx, () => {
                ctx.send([{ pointer: ptr, op: 'delete' }]);
            });
            const current = isObject(item) ? item[tag] : undefined;
            const variant = schema.variants.find((v) => v.tag_value === current);
            const select = el('select');
            select.dataset.pointer = child(ptr, tag);
            if (!variant) {
                select.append(el('option', '', current === undefined ? '' : String(current)));
            }
            for (const v of schema.variants) {
                const option = el('option', '', v.tag_value);
                option.value = v.tag_value;
                select.append(option);
            }
            select.value = variant ? variant.tag_value : select.options[0].value;
            // a new kind keeps the keys of the entry declared by the new variant
            select.addEventListener('change', () => {
                const next = schema.variants.find((v) => v.tag_value === select.value);
                if (!next) {
                    return;
                }
                const kept = { [tag]: next.tag_value, ...presentKeys(item, next.fields) };
                ctx.send([setEdit(ptr, entryText(kept, next.fields), ctx)]);
            });
            head.insertBefore(select, head.lastChild);
            card.append(head);
            if (variant) {
                const fields = variant.fields.filter((f) => f.name !== tag);
                card.append(renderGrid(fields, [{ pointer: ptr }], ctx));
            } else {
                card.append(preOf(item));
            }
            wrap.append(card);
        });
        const addRow = el('div', 'add-row');
        addRow.append(iconButton('+', 'add an entry', () => {
            const text = JSON.stringify({ [tag]: 'list', field: '', values: [] });
            ctx.send([insertEdit(base, text, ctx)]);
        }, 'add'));
        wrap.append(addRow);
        return wrap;
    }

    function renderOptionalStructDict(section, ctx) {
        const base = pointer([section.name]);
        const mapping = valueAt(ctx.tree, base);
        const table = isObject(mapping) ? mapping : {};
        const wrap = sectionShell(section, ctx);
        const known = new Set();
        for (const field of section.schema.fields) {
            known.add(field.name);
            const ptr = child(base, field.name);
            const enabled = table[field.name] !== undefined && table[field.name] !== null;
            const item = el('div', 'measurement');
            const toggle = el('label', 'measurement-toggle');
            const box = el('input');
            box.type = 'checkbox';
            box.checked = enabled;
            box.dataset.pointer = ptr;
            box.addEventListener('change', () => {
                if (box.checked) {
                    ctx.send([setEdit(ptr, '{}', ctx)]);
                } else {
                    ctx.send([{ pointer: ptr, op: 'delete' }]);
                }
            });
            toggle.append(box, el('span', 'measurement-name', field.name));
            addTitle(toggle, fieldTip(field));
            if (enabled) {
                markProvenance(toggle, ptr, ctx);
                markErrors(toggle, ptr, ctx);
            }
            item.append(toggle);
            if (enabled && field.fields.length) {
                item.append(renderGrid(field.fields, [{ pointer: ptr }], ctx));
            }
            wrap.append(item);
        }
        for (const key of Object.keys(table)) {
            if (known.has(key)) {
                continue;
            }
            const ptr = child(base, key);
            const card = el('div', 'entry');
            card.append(entryHeader(ptr, key, ctx, () => {
                ctx.send([{ pointer: ptr, op: 'delete' }]);
            }));
            card.append(preOf(table[key]));
            wrap.append(card);
        }
        return wrap;
    }

    function renderDict(section, ctx) {
        const base = pointer([section.name]);
        const mapping = valueAt(ctx.tree, base);
        const entries = isObject(mapping) ? mapping : {};
        const item = section.schema;
        const wrap = sectionShell(section, ctx);
        const structLike = item.kind === 'struct' && !isReadOnly(item);
        for (const key of Object.keys(entries)) {
            const ptr = child(base, key);
            const card = el('div', 'entry dict-entry');
            card.append(entryHeader(ptr, key, ctx, () => {
                ctx.send([{ pointer: ptr, op: 'delete' }]);
            }));
            if (structLike) {
                card.append(renderGrid(item.fields, [{ pointer: ptr }], ctx));
            } else {
                const table = el('table', 'spec-grid');
                const tr = el('tr');
                tr.append(valueCell(item, ptr, ctx));
                table.append(tr);
                card.append(table);
            }
            wrap.append(card);
        }
        const addRow = el('div', 'add-row');
        const keyInput = el('input');
        keyInput.type = 'text';
        keyInput.placeholder = 'new key';
        keyInput.spellcheck = false;
        const add = () => {
            const key = keyInput.value.trim();
            if (key === '' || key in entries) {
                return;
            }
            ctx.send([setEdit(child(base, key), structLike ? '{}' : 'null', ctx)]);
        };
        keyInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                add();
            }
        });
        addRow.append(keyInput, iconButton('+', 'add the key', add, 'add'));
        wrap.append(addRow);
        return wrap;
    }

    function renderOpaque(section, ctx) {
        const wrap = sectionShell(section, ctx);
        const value = valueAt(ctx.tree, pointer([section.name]));
        wrap.append(preOf(value === undefined ? section.field.default : value));
        wrap.append(el('p', 'placeholder', 'this section is edited in the file'));
        return wrap;
    }
}
