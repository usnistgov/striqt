from .radio import base
from .structs import Sweep, RadioCapture

import labbench as lb
from channel_analysis import waveform
import xarray as xr
from functools import lru_cache
import msgspec
import numpy as np
from frozendict import frozendict
import pandas as pd

CAPTURE_DIM = 'capture'
TIMESTAMP_NAME = 'timestamp'

FIELD_ATTRS = {
    RadioCapture.center_frequency.__name__: {
        'label': 'RF center frequency',
        'units': 'Hz',
    },
    RadioCapture.channel.__name__: {'label': 'RX hardware input port'},
    RadioCapture.gain.__name__: {
        'label': 'internal gain setting inside the radio',
        'unit': 'dB',
    },
    RadioCapture.duration.__name__: {'label': 'duration of the capture', 'unit': 's'},
    RadioCapture.sample_rate.__name__: {
        'label': 'sample rate of the waveform',
        'unit': 'S/s',
    },
    RadioCapture.analysis_bandwidth.__name__: {
        'label': 'filtered bandwidth of the received waveform',
        'unit': 'Hz',
    },
    RadioCapture.lo_shift.__name__: {
        'label': 'direction of the LO shift (or None for no shift)',
        'unit': 'Hz',
    },
    RadioCapture.preselect_if_frequency.__name__: {
        'label': 'IF filter center frequency',
        'unit': 'Hz',
    },
    RadioCapture.preselect_lo_gain.__name__: {
        'label': 'gain of the LO stage',
        'unit': 'dB',
    },
    RadioCapture.preselect_rf_gain.__name__: {
        'label': 'preselector gain setting',
        'unit': 'dB',
    },
    TIMESTAMP_NAME: {'label': 'Capture start time'},
}

@lru_cache
def _capture_coord_template(sweep_fields: tuple[str, ...]):
    capture = RadioCapture()
    coords = {}

    for field in sweep_fields:
        coords[field] = xr.Variable(
            (CAPTURE_DIM,),
            [getattr(capture, field)],
            fastpath=True
        )

    coords[TIMESTAMP_NAME] = xr.Variable((CAPTURE_DIM,), [pd.Timestamp('now')], fastpath=True)

    return xr.Coordinates(coords)

def capture_to_coords(capture: RadioCapture, sweep_fields: list[str], timestamp=None):
    coords = _capture_coord_template(sweep_fields).copy(deep=True)

    for field in sweep_fields:
        coords[field].values[:] = [getattr(capture, field)]

    if timestamp is not None:
        coords[TIMESTAMP_NAME].values[:] = [timestamp]

    return coords

def sweep(
    radio: base.RadioDevice, run_spec: Sweep, sweep_fields: list[str]
) -> xr.Dataset:
    data = []
    spec = run_spec.channel_analysis
    sweep_fields = tuple(sweep_fields)

    timestamps = []

    attrs = {}

    for capture in run_spec.captures:
        # treat swept fields as coordinates/indices
        desc = ', '.join([f'{k}={v}' for k, v in msgspec.to_builtins(capture).items()])

        with lb.stopwatch(f'{desc}: '):
            radio.arm(capture)
            iq, timestamp = radio.acquire()
            coords = capture_to_coords(capture, sweep_fields, timestamp=timestamp)
            analysis = (
                waveform
                .analyze_by_spec(iq, capture, spec=spec)
                .assign_coords(coords)
            )

        # remove swept fields from the metadata
        for f in sweep_fields:
            del analysis.attrs[f]

        data.append(analysis)

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(sweep_fields) + (TIMESTAMP_NAME,):
        ds[k].attrs.update(FIELD_ATTRS[k])

    return ds