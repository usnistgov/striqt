from .radio import base
from .structs import Sweep, RadioCapture, get_attrs

import labbench as lb
from channel_analysis import waveform
import xarray as xr
from functools import lru_cache
import msgspec
import numpy as np
from frozendict import frozendict
import pandas as pd
import typing

CAPTURE_DIM = 'capture'
TIMESTAMP_NAME = 'timestamp'

@lru_cache
def _capture_coord_template(sweep_fields: tuple[str, ...]):
    capture = RadioCapture()
    coords = {}

    for field in sweep_fields:
        coords[field] = xr.Variable(
            (CAPTURE_DIM,), [getattr(capture, field)], fastpath=True
        )

    coords[TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,), [pd.Timestamp('now')], fastpath=True
    )

    return xr.Coordinates(coords)


def capture_to_coords(capture: RadioCapture, sweep_fields: list[str], timestamp=None):
    coords = _capture_coord_template(sweep_fields).copy(deep=True)

    for field in sweep_fields:
        coords[field].values[:] = [getattr(capture, field)]

    if timestamp is not None:
        coords[TIMESTAMP_NAME].values[:] = [timestamp]

    return coords


def sweep(
    radio: base.RadioBase, sweep: Sweep, swept_fields: list[str]
) -> xr.Dataset:
    data = []
    spec = sweep.channel_analysis
    swept_fields = tuple(swept_fields)

    for capture in sweep.captures:
        # treat swept fields as coordinates/indices
        desc = ', '.join([f'{k}={v}' for k, v in msgspec.to_builtins(capture).items()])

        with lb.stopwatch(f'{desc}: '):
            radio.arm(capture)
            iq, timestamp = radio.acquire()
            coords = capture_to_coords(capture, swept_fields, timestamp=timestamp)
            analysis = waveform.analyze_by_spec(iq, capture, spec=spec).assign_coords(
                coords
            )

        # remove swept fields from the metadata
        for f in swept_fields:
            del analysis.attrs[f]

        data.append(analysis)

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(swept_fields):
        ds[k].attrs.update(get_attrs(RadioCapture, k))
    ds[TIMESTAMP_NAME].attrs.update(label='Capture start time')

    return ds