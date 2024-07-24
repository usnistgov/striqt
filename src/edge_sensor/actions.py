from .radio import base
from .structs import Sweep, RadioCapture, get_attrs

import labbench as lb
import xarray as xr
import msgspec
import numpy as np
import pandas as pd
import typing
from channel_analysis import waveform
from functools import lru_cache

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


def sweep(radio: base.RadioBase, sweep: Sweep, swept_fields: list[str]) -> xr.Dataset:
    data = []
    spec = sweep.channel_analysis
    swept_fields = tuple(swept_fields)

    radio.arm(sweep.captures[0])

    for i, capture in enumerate(sweep.captures):
        # treat swept fields as coordinates/indices
        desc = ', '.join([f'{k}={getattr(capture, k)}' for k in swept_fields])

        with lb.stopwatch(f'{desc}: '):
            iq, timestamp = radio.acquire()
            # prepare the next capture while we analyze
            if i + 1 < len(sweep.captures):
                radio.arm(sweep.captures[i + 1])
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
