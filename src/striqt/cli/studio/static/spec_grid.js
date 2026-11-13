// the grid of the Spec tab: the rows of a schema, the provenance and validation
// marks, the value cells and their controls, and the per-port cells of a capture.
//
// Every function takes the `ctx` built by spec.js: `tree`, `origin`, `regions`,
// `files`, `errors`, `editable`, `changed`, `busy`, `review`, `schema`, `send`, and
// `revert(section)`.

import { el, addTitle, iconButton, preOf } from './util.js';
import {
    pointer, child, parentPointer, valueAt, shortName, scalarText, flowText, displayText,
    typedText, setEdit, commit, resizePorts,
} from './spec_yaml.js';

export const READ_ONLY_GLYPH = '\u{1F512}';

// %% schema fields

export function fieldTip(field) {
    return field.description || field.standard_name || '';
}

export function isReadOnly(field) {
    return field.read_only
        || field.kind === 'opaque'
        || field.kind === 'dict'
        || (field.kind === 'struct' && (field.variants.length > 0 || field.fields.length === 0));
}

// the rows of a grid: one per field, with the rows of a nested struct after its own
// row; `owner` is the owner of the top-level field for the group label
export function fieldRows(fields, path = [], depth = 0, owner = null) {
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

// %% provenance and validation marks

// the file index that holds `ptr`, or that would receive a new key at `ptr`: the
// last file of the nearest region, else the origin of the nearest ancestor
export function fileOf(ptr, ctx) {
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

export function markProvenance(node, ptr, ctx) {
    const index = fileOf(ptr, ctx);
    if (index === null) {
        return;
    }
    node.dataset.file = String(index);
    const region = ctx.regions[ptr];
    if (region && region.length > 1) {
        addTitle(node, 'merged from:\n' + region.map((i) => ctx.files[i]).join('\n'));
    } else {
        addTitle(node, ctx.files[index]);
    }
}

// the messages of the errors at `ptr` or under it; at `ptr` only when `exact`
export function errorsAt(ptr, ctx, exact = false) {
    const found = [];
    for (const error of ctx.errors) {
        const where = [error.path, ...(error.locations || [])];
        if (where.some((w) => w && (w === ptr || (!exact && w.startsWith(`${ptr}/`))))) {
            found.push(error.message);
        }
    }
    return found;
}

export function markErrors(node, ptr, ctx, exact = false) {
    const found = errorsAt(ptr, ctx, exact);
    if (found.length) {
        node.classList.add('invalid');
        addTitle(node, found.join('\n'));
    }
}

// %% sections and grids

export function sectionShell(section, ctx) {
    const wrap = el('section', 'spec-section');
    const header = el('header', 'section-header');
    const ptr = pointer([section.name]);
    header.append(el('h2', '', section.name));
    if (ctx.changed.has(section.name) && ctx.editable) {
        const revert = el('button', 'revert-button button outline', 'Revert');
        revert.type = 'button';
        revert.title = 'discard the edits to this section since the last apply';
        revert.disabled = ctx.busy;
        revert.addEventListener('click', () => ctx.revert(section.name));
        header.append(revert);
    }
    if (section.read_only) {
        header.append(el('span', 'lock-glyph', READ_ONLY_GLYPH), el('span', 'note', 'read-only'));
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
export function renderGrid(fields, columns, ctx, options = {}) {
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
                const custom = row.depth === 0 && options.controls
                    ? options.controls[row.field.name] || null
                    : null;
                const td = valueCell(row.field, ptr, ctx, custom);
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

// `custom` replaces the default control with `custom(field, ptr, value, present, ctx)`
export function valueCell(field, ptr, ctx, custom = null) {
    const td = el('td', 'value');
    const { value, present } = decorateCell(td, ptr, ctx);
    if (custom) {
        td.append(custom(field, ptr, value, present, ctx));
    } else if (isReadOnly(field)) {
        td.classList.add('read-only');
        td.append(preOf(present ? value : field.default));
    } else {
        td.append(control(field, ptr, value, present, ctx));
    }
    return td;
}

export function textInput(ptr, text) {
    const input = el('input', 'text-input');
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
        const select = el('select', 'text-input');
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

export function entryHeader(ptr, label, ctx, onRemove) {
    const head = el('div', 'entry-header');
    head.append(el('span', 'entry-label', label));
    markProvenance(head, ptr, ctx);
    markErrors(head, ptr, ctx);
    if (onRemove) {
        head.append(iconButton('×', `remove ${label}`, onRemove, 'remove'));
    }
    return head;
}
