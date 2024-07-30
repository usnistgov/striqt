from __future__ import annotations
from .radio import base, soapy
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


def single_analysis(iq, timestamp, radio, capture, swept_fields, spec):
    with lb.stopwatch('analyze', logger_level='debug'):
        iq = radio.resampling_correction(iq, capture)
        coords = capture_to_coords(capture, swept_fields, timestamp=timestamp)

        analysis = waveform.analyze_by_spec(iq, capture, spec=spec).assign_coords(coords)

        for f in swept_fields:
            del analysis.attrs[f]

        return analysis


def sweep(
    radio: base.RadioBase, sweep: Sweep, swept_fields: list[str]
) -> xr.Dataset | None:
    data = []
    spec = sweep.channel_analysis
    swept_fields = tuple(swept_fields)

    if len(sweep.captures) == 0:
        return None

    iq = None
    t = None

    prevs = [None] + sweep.captures
    currs = sweep.captures + [None]
    nexts = sweep.captures[1:] + [None,None]

    for (prev, curr, next_) in zip(prevs, currs, nexts):
        if curr is None:
            desc = 'last analysis'
        else:
            # treat swept fields as coordinates/indices
            desc = ', '.join([f'{k}={getattr(curr, k)}' for k in swept_fields])

        with lb.stopwatch(f'{desc}: '):
            calls = {}

            if curr is not None:
                # skip at end to allow the final analysis
                calls['acquire'] = lb.Call(radio.acquire, curr, next_capture=next_, correction=False)

            if prev is not None:
                # skip the first iteration before any iq data is available
                calls['analyze'] = lb.Call(single_analysis, iq, t, radio, prev, swept_fields, spec)

            ret = lb.concurrently(**calls, flatten=False)

        
        if 'analyze' in ret:
            data.append(ret['analyze'])

        if 'acquire' in ret:
            iq, t = ret['acquire']

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(swept_fields):
        ds[k].attrs.update(get_attrs(RadioCapture, k))
    ds[TIMESTAMP_NAME].attrs.update(label='Capture start time')

    return ds
