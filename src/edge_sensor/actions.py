from __future__ import annotations
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


def arm_acquire(radio, capture) -> tuple:
    if capture is None:
        return None, None
    radio.arm(capture)
    return radio.acquire(capture)


def sweep(
    radio: base.RadioBase, sweep: Sweep, swept_fields: list[str]
) -> xr.Dataset | None:
    data = []
    spec = sweep.channel_analysis
    swept_fields = tuple(swept_fields)

    if len(sweep.captures) == 0:
        return None

    capture = sweep.captures[0]
    radio.arm(capture)
    iq_fast, t = radio.acquire(capture)

    # Tomorrow-Dan: shift arm() to occur during the analysis
    for capture, next_capture in zip(sweep.captures, list(sweep.captures[1:] + [None])):
        # treat swept fields as coordinates/indices
        desc = ', '.join([f'{k}={getattr(capture, k)}' for k in swept_fields])

        with lb.stopwatch(f'{desc}: '):
            iq = radio.resampling_correction(iq_fast, capture)
            if next_capture is not None:
                radio.arm(next_capture)
            # iq.get()

            if next_capture is None:
                coords = capture_to_coords(capture, swept_fields, timestamp=t)
                analysis = waveform.analyze_by_spec(iq, capture, spec=spec).assign_coords(coords)
            else:
                # results = waveform._evaluate_raw_channel_analysis(iq, capture, spec=spec)
                coords = capture_to_coords(capture, swept_fields, timestamp=t)
                # iq_fast, t = radio.acquire(capture)
                # analysis = (
                #     waveform._package_channel_analysis(capture, results)
                #     .assign_coords(coords)
                # )
                with lb.stopwatch('concurrent'):
                    ret = lb.concurrently(
                        lb.Call(radio.acquire, next_capture),
                        lb.Call(waveform.analyze_by_spec, iq, capture, spec=spec),
                    )

                analysis = ret['analyze_by_spec'].assign_coords(coords)
                iq_fast, t = ret['acquire']
                # iq_fast.get()

            # print(ret.keys())
            # iq_fast, timestamp = arm_acquire(radio, next_capture)
            # analysis = waveform.analyze_by_spec(iq, capture, spec=spec).assign_coords(
            #     coords
            # )

        # remove swept fields from the metadata
        for f in swept_fields:
            del analysis.attrs[f]

        data.append(analysis)

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(swept_fields):
        ds[k].attrs.update(get_attrs(RadioCapture, k))
    ds[TIMESTAMP_NAME].attrs.update(label='Capture start time')

    return ds
