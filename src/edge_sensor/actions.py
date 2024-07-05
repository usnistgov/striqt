from .radio import base
from .structs import Sweep, RadioCapture

import labbench as lb
from channel_analysis import waveform
import xarray as xr
from functools import cache

FIELD_ATTRS = {
    'center_frequency': {
        'label': 'RF center frequency',
        'units': 'Hz',
    },
    'channel': {
        'label': 'RX hardware input port'
    },
    'gain': {
        'label': 'internal gain setting inside the radio',
        'unit': 'dB'
    },
    'duration': {
        'label': 'duration of the capture',
        'unit': 's'
    },
    'sample_rate': {
        'label': 'sample rate of the waveform',
        'unit': 'S/s'
    },
    'analysis_bandwidth': {
        'label': 'filtered bandwidth of the received waveform',
        'unit': 'Hz',
    },
    'lo_shift': {
        'label': 'direction of the LO shift (or None for no shift)',
        'unit': 'Hz'
    },
    'preselect_if_frequency': {
        'label': 'IF filter center frequency',
        'unit': 'Hz',
    },
    'preselect_lo_gain': {
        'label': 'gain of the LO stage',
        'unit': 'dB'
    },
    'preselect_rf_gain': {
        'label': 'preselector gain setting',
        'unit': 'dB'
    }
}

@cache
def _template_coordinates(fields):
    coords = xr.Coordinates(
        {f: [getattr(RadioCapture, f)] for f in fields}
    )

    for field in fields:
        coords[field].attrs = FIELD_ATTRS[field]

    return coords

def coordinates(capture: RadioCapture, fields):
    fields = tuple(fields)
    coords = _template_coordinates(fields).copy()
    for field in fields:
        coords[field].values[:] = [getattr(capture, field)]
    return coords

def sweep(radio: base.RadioDevice, run_spec: Sweep, sweep_fields: list[str]) -> xr.Dataset:
    data = []

    for capture in run_spec.captures:
        # treat swept fields as coordinates/indices
        coords = coordinates(capture, sweep_fields)
        desc = ', '.join([f'{k}={v[0]}' for k,v in coords.items()])

        with lb.stopwatch(f'{desc}: '):
            radio.arm(capture)
            iq, timestamp = radio.acquire()
            coords['timestamp'] = [timestamp]
            analysis = (
                waveform
                .analyze_by_spec(iq, capture, spec=run_spec.channel_analysis)
                .assign_coords(coords)
            )

        # remove swept fields from the metadata
        for f in sweep_fields:
            del analysis.attrs[f]

        data.append(analysis)

    return xr.combine_by_coords(data)