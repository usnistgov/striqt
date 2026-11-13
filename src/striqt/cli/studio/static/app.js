// the page shell: the websocket client, the message router, and the tab bar.
//
// Messages from the server are dispatched by `type` to the handlers that the tab
// modules register with `on(type, handler)`. The latest `state`, `schema`, and
// `spec` messages are kept in `latest` so a tab opened later can render them.

import { init as initPreview } from './preview.js';
import { init as initSpec } from './spec.js';
import { init as initRun } from './run.js';

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const statusButton = document.getElementById('status-button');
const modeBadge = document.getElementById('mode-badge');
const headerTitle = document.getElementById('header-title');

let ws = null;
// set by the status button; suppresses the automatic reconnect
let userDisconnected = false;

const handlers = new Map();
const tabListeners = new Map();

export const latest = { state: null, schema: null, spec: null };

export function send(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    }
}

export function on(type, handler) {
    if (!handlers.has(type)) {
        handlers.set(type, []);
    }
    handlers.get(type).push(handler);
}

export function connected() {
    return !!ws && ws.readyState === WebSocket.OPEN;
}

function dispatch(msg) {
    if (msg.type in latest) {
        latest[msg.type] = msg;
    }
    if (msg.type === 'state') {
        modeBadge.textContent = msg.mode;
        modeBadge.dataset.mode = msg.mode;
        headerTitle.textContent = '\u{1F4E1} ' + msg.spec_filename;
    }
    if (msg.type === 'error') {
        console.error('Server error:', msg.message);
    }
    for (const handler of handlers.get(msg.type) || []) {
        try {
            handler(msg);
        } catch (e) {
            console.error(`handler for ${msg.type} failed`, e);
        }
    }
}

function setConnectionState(state) {
    statusDot.classList.remove('connecting', 'connected');
    if (state === 'connecting') {
        statusDot.classList.add('connecting');
        statusText.textContent = 'Connecting...';
        statusButton.title = 'Cancel';
    } else if (state === 'connected') {
        statusDot.classList.add('connected');
        statusText.textContent = 'Connected';
        statusButton.title = 'Disconnect';
    } else {
        statusText.textContent = 'Disconnected';
        statusButton.title = 'Connect';
    }
    document.body.dataset.connection = state;
    for (const handler of handlers.get('connection') || []) {
        handler(state);
    }
}

function connect() {
    userDisconnected = false;
    setConnectionState('connecting');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        setConnectionState('connected');
        send({ type: 'get_state' });
    };
    ws.onclose = () => {
        setConnectionState('disconnected');
        if (!userDisconnected) {
            setTimeout(connect, 2000);
        }
    };
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    ws.onmessage = (event) => {
        dispatch(JSON.parse(event.data));
    };
}

statusButton.addEventListener('click', () => {
    if (ws && ws.readyState !== WebSocket.CLOSED) {
        userDisconnected = true;
        ws.close();
    } else {
        connect();
    }
});

// %% tabs

export function currentTab() {
    return document.querySelector('.tab.active').dataset.tab;
}

export function selectTab(name) {
    for (const button of document.querySelectorAll('.tab')) {
        button.classList.toggle('active', button.dataset.tab === name);
    }
    for (const panel of document.querySelectorAll('.tab-panel')) {
        panel.hidden = panel.id !== `tab-${name}`;
    }
    for (const listener of tabListeners.get(name) || []) {
        listener();
    }
}

// `callback` runs each time the tab named `name` is shown
export function onTab(name, callback) {
    if (!tabListeners.has(name)) {
        tabListeners.set(name, []);
    }
    tabListeners.get(name).push(callback);
}

for (const button of document.querySelectorAll('.tab')) {
    button.addEventListener('click', () => selectTab(button.dataset.tab));
}

const app = { send, on, onTab, selectTab, currentTab, latest, connected };
initPreview(app);
initSpec(app);
initRun(app);

connect();
