"""shared setup of the `striqt.cli.studio` tests: a looped single-tone sweep and a
session over it"""

from __future__ import annotations

import threading

from synthetic_sources import (
    SPECTROGRAM,
    TONE_OFFSETS,
    preset_sweep,
    run_in_memory,
    tone_captures,
)

import striqt.sensor as ss
from striqt.cli.studio import session
from striqt.sensor.specs import sequencing

SNR_LOOP = ss.specs.List(field='snr', values=(10.0, 20.0))


def looped_sweep(analysis=SPECTROGRAM):
    """return a single-tone sweep with a `Repeat` and the `snr` loop"""
    return preset_sweep(
        'single_tone',
        tone_captures(TONE_OFFSETS),
        analysis=analysis,
        loops=(ss.specs.Repeat(count=2), SNR_LOOP),
    )


def open_spec(sweep, **overrides) -> session.OpenSpec:
    """return the `OpenSpec` of `sweep` with no source, for tests without resources"""
    from striqt.figures.backend import PlotlyBackend

    fields = {
        'sweep': sweep,
        'source_id': '',
        'choices': sequencing.plan_choices(sweep, None),
        'varying_fields': frozenset(),
        'adjustments': {},
        'sink_path': None,
        'available_variables': (),
        'variable_labels': {},
        'plotter': PlotlyBackend(session._default_plot_options().plotter),
        'data_select': {},
    }
    fields.update(overrides)
    return session.OpenSpec(**fields)


def preview_session(sweep, **overrides) -> session.StudioSession:
    """return a session over `sweep` with the first entry of every choice selected"""
    state = session.StudioSession(open=open_spec(sweep, **overrides))
    session.select_first(state)
    return state


def analyzed_dataset():
    """return the `DelayedDataset` of one spectrogram capture"""
    sweep = preset_sweep(
        'single_tone', tone_captures(TONE_OFFSETS[:1]), analysis=SPECTROGRAM
    )
    return run_in_memory(sweep, take=1, delayed=True)[0]


def recording_events(state, on_data=None):
    """return `SessionEvents` that record each mode change and each dataset of `state`,
    with the lists and the close event the events fill; `on_data` is called after
    each dataset is recorded"""
    import logging
    import threading

    modes = []
    datasets = []
    closed = threading.Event()

    def data_ready():
        datasets.append((state.mode, state.dataset))
        if on_data is not None:
            on_data()

    events = session.SessionEvents(
        state_changed=lambda: modes.append(state.mode),
        data_ready=data_ready,
        attach_logs=logging.NullHandler,
        detach_logs=lambda handler: None,
        closed=closed.set,
    )
    return events, modes, datasets, closed


class HookedStop(threading.Event):
    """an Event with a `wait` that calls ``on_wait(self)`` and returns without
    blocking"""

    def __init__(self, on_wait):
        super().__init__()
        self.on_wait = on_wait
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        self.on_wait(self)
        return self.is_set()
