// the Spec tab: the section list, the toolbar, the messages, and the section
// renderers over the editor schema and the merged spec tree.
//
// A `schema`, `spec`, or `state` message re-renders the grid only while the tab is
// shown; otherwise the tab is marked dirty and renders when it is next shown.

import { el, isObject, addTitle, iconButton, preOf, busy } from './util.js';
import {
    pointer, child, parentPointer, valueAt, shortName, presentKeys, scalarText, entryText,
    portCount, setEdit, insertEdit,
} from './spec_yaml.js';
import {
    READ_ONLY_GLYPH, fieldTip, isReadOnly, errorsAt, markProvenance, markErrors, sectionShell,
    renderGrid, valueCell, entryHeader,
} from './spec_grid.js';

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
    let lastValidation = null;
    // pointers of per-port fields resized to a new port count, until edited
    const review = new Set();

    app.on('schema', scheduleRender);
    app.on('spec', scheduleRender);
    app.on('state', (msg) => {
        renderActions();
        const key = JSON.stringify([
            msg.validation, msg.editable, msg.changed_sections || [], msg.mode,
            msg.apply_error,
        ]);
        if (key !== lastValidation) {
            lastValidation = key;
            scheduleRender();
        } else {
            renderMessages(context());
        }
    });
    app.on('connection', renderActions);
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
            busy: !!state && busy(state.mode),
            applyError: state ? state.apply_error || null : null,
            review,
            schema: app.latest.schema ? app.latest.schema.schema : null,
            send: sendEdits,
            revert: (section) => app.send({ type: 'revert', section }),
        };
    }

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
            body.append(el('p', 'note warn', 'JSON specs are read-only in the page'));
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
        const title = el('div', 'spec-nav-title panel-heading', shortName(schema.action_cls));
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
            actions.append(el('span', 'note warn', 'JSON specs are read-only in the page'));
        }
        if (state.applied_differs) {
            actions.append(el('span', 'note warn', 'edited spec differs from the applied one'));
        }
        const apply = el('button', 'apply-button button', 'Apply');
        apply.type = 'button';
        const blocked = busy(state.mode);
        apply.disabled = !(state.validation.ok && state.editable && app.connected() && !blocked);
        apply.title = blocked ? 'stop the run before applying the spec'
            : !state.validation.ok ? 'the spec files do not validate'
                : 'apply the edited spec to the acquisition';
        apply.addEventListener('click', () => app.send({ type: 'apply' }));
        actions.append(apply);
    }

    function renderMessages(ctx) {
        messages.innerHTML = '';
        const texts = ctx.errors.map((e) => (e.path ? `${e.path}: ` : '') + e.message);
        if (ctx.applyError) {
            texts.unshift(`apply failed: ${ctx.applyError.split('\n').filter(Boolean).pop()}`);
        }
        messages.hidden = texts.length === 0;
        for (const text of texts) {
            messages.append(el('div', 'message', text));
        }
    }
}

// %% the sections

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

// the rows of a loop variant: `isin` then `field` before the rest in schema order;
// a `repeat` loop has no field, so neither row is shown
function loopRows(variant, tag) {
    const fields = variant.fields.filter((f) => f.name !== tag);
    if (variant.tag_value === 'repeat') {
        return fields.filter((f) => f.name !== 'field' && f.name !== 'isin');
    }
    const first = ['isin', 'field']
        .map((name) => fields.find((f) => f.name === name))
        .filter(Boolean);
    return [...first, ...fields.filter((f) => !first.includes(f))];
}

// the names a loop may step: the editable capture fields other than `port` when
// `isin` is `capture`, else the parameter names of the enabled measurements
function loopFieldOptions(isin, ctx) {
    const sections = ctx.schema ? ctx.schema.sections : [];
    const names = new Set();
    if (isin === 'analysis') {
        const analysis = sections.find((s) => s.kind === 'optional_struct_dict');
        const enabled = analysis ? valueAt(ctx.tree, pointer([analysis.name])) : undefined;
        for (const measurement of analysis ? analysis.schema.fields : []) {
            const value = isObject(enabled) ? enabled[measurement.name] : undefined;
            if (value === undefined || value === null) {
                continue;
            }
            for (const f of measurement.fields) {
                names.add(f.name);
            }
        }
        return [...names].sort();
    }
    const captures = sections.find((s) => s.name === 'captures' && s.kind === 'struct_list');
    for (const f of captures ? captures.schema.fields : []) {
        if (f.name !== 'port' && !['opaque', 'dict', 'struct'].includes(f.kind)) {
            names.add(f.name);
        }
    }
    return [...names];
}

// the `field` control of a loop: a select over `loopFieldOptions` for the loop's
// `isin`, with the current name kept as an option when it is not in the list
function loopFieldControl(item) {
    return (field, ptr, value, present, ctx) => {
        const isin = isObject(item) && item.isin !== undefined ? item.isin : 'capture';
        const options = loopFieldOptions(isin, ctx);
        const select = el('select', 'text-input');
        select.dataset.pointer = ptr;
        if (!present) {
            select.append(el('option', '', ''));
        } else if (!options.includes(String(value))) {
            select.append(el('option', '', String(value)));
        }
        for (const name of options) {
            select.append(el('option', '', name));
        }
        select.value = present ? String(value) : '';
        select.addEventListener('change', () => {
            if (select.value !== '') {
                ctx.send([setEdit(ptr, scalarText(select.value, 'string'), ctx)]);
            }
        });
        return select;
    };
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
        // a Repeat is the outermost loop, so only loop 0 is offered `repeat`; a
        // later loop that is `repeat` keeps its value in the list to be changed
        const offered = schema.variants.filter((v) => i === 0 || v.tag_value !== 'repeat');
        const select = el('select');
        select.dataset.pointer = child(ptr, tag);
        if (!variant) {
            select.append(el('option', '', current === undefined ? '' : String(current)));
        } else if (!offered.includes(variant)) {
            const option = el('option', '', variant.tag_value);
            option.value = variant.tag_value;
            select.append(option);
        }
        for (const v of offered) {
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
            card.append(renderGrid(loopRows(variant, tag), [{ pointer: ptr }], ctx, {
                controls: { field: loopFieldControl(item) },
            }));
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
    const keyInput = el('input', 'text-input');
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
