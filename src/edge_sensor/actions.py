from .radio import base
from .structs import Sweep, RadioCapture

import labbench as lb
from channel_analysis import waveform
import xarray as xr
from functools import cache
import msgspec
import numpy as np

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
    'timestamp': {'label': 'Capture start time'},
}


def sweep(
    radio: base.RadioDevice, run_spec: Sweep, sweep_fields: list[str]
) -> xr.Dataset:
    data = []
    spec = run_spec.channel_analysis

    for capture in run_spec.captures:
        # treat swept fields as coordinates/indices
        coords = {k: [getattr(capture, k)] for k in sweep_fields}
        desc = ', '.join([f'{k}={v[0]}' for k, v in coords.items()])

        with lb.stopwatch(f'{desc}: '):
            radio.arm(capture)
            iq, timestamp = radio.acquire(capture.channel)
            coords['timestamp'] = [timestamp]
            analysis = waveform.analyze_by_spec(iq, capture, spec=spec).assign_coords(
                coords
            )

        # remove swept fields from the metadata
        for f in sweep_fields:
            del analysis.attrs[f]

        data.append(analysis)

    ds = xr.combine_by_coords(data)
    for k in sweep_fields:
        ds[k].attrs = FIELD_ATTRS[k]

    return ds
