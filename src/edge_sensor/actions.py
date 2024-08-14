from __future__ import annotations
from .radio import RadioDevice
from .structs import Sweep, RadioCapture, get_attrs, to_builtins
from .util import zip_offsets
from . import iq_corrections
from channel_analysis.structs import ChannelAnalysis

import labbench as lb
from iqwaveform.util import Array
import xarray as xr
import pandas as pd
from channel_analysis import waveform
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Generator, Any
import pickle

CAPTURE_DIM = 'capture'
TIMESTAMP_NAME = 'timestamp'


@lru_cache
def _capture_coord_template(sweep_fields: tuple[str, ...]):
    """returns a valid cached xarray coordinate for the given list of swept fields.

    the
    """

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


@dataclass
class _RadioCaptureAnalyzer:
    """an IQ data analysis/packaging manager given a radio and desired channel analyses"""

    __name__ = 'analyze'

    radio: RadioDevice
    analysis_spec: list[ChannelAnalysis]
    remove_attrs: Optional[tuple[str, ...]] = None
    extra_attrs: Optional[dict[str, Any]] = None
    calibration: Optional[xr.Dataset] = None

    def __call__(self, iq: Array, timestamp, capture: RadioCapture) -> xr.Dataset:
        """analyze iq from a capture and package it into a dataset"""

        with lb.stopwatch('analyze', logger_level='debug'):
            # for performance, GPU operations are all here in the same thread
            iq = iq_corrections.resampling_correction(iq, capture, self.radio, force_calibration=self.calibration)
            coords = self.get_coords(capture, timestamp=timestamp)

            analysis = waveform.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            ).assign_coords(coords)

        if self.remove_attrs is not None:
            for f in self.remove_attrs:
                del analysis.attrs[f]

        for k in tuple(self.remove_attrs):
            analysis[k].attrs.update(get_attrs(RadioCapture, k))

        if self.extra_attrs is not None:
            analysis.attrs.update(self.extra_attrs)

        analysis[TIMESTAMP_NAME].attrs.update(label='Capture start time')

        return analysis

    def __post_init__(self):
        if self.remove_attrs is not None:
            self.remove_attrs = tuple(self.remove_attrs)

    def get_coords(self, capture: RadioCapture, timestamp):
        coords = _capture_coord_template(self.remove_attrs).copy(deep=True)

        for field in self.remove_attrs:
            value = getattr(capture, field)
            if isinstance(value, str):
                # to coerce strings as variable-length types later for storage
                coords[field] = coords[field].astype('object')
            coords[field].values[:] = [value]

        if timestamp is not None:
            coords[TIMESTAMP_NAME].values[:] = [timestamp]

        return coords


def describe_capture(capture: RadioCapture, swept_fields):
    return ', '.join([f'{k}={getattr(capture, k)}' for k in swept_fields])


def iter_sweep(
    radio: RadioDevice, sweep: Sweep, swept_fields: list[str], calibration: xr.Dataset=None
) -> Generator[xr.Dataset]:
    """iterate through sweep captures on the specified radio, yielding a dataset for each"""

    attrs = {
        # metadata fields
        'radio_id': radio.id,
        'radio_setup': to_builtins(sweep.radio_setup),
        'description': to_builtins(sweep.description),
    }

    analyze = _RadioCaptureAnalyzer(
        radio=radio,
        analysis_spec=sweep.channel_analysis,
        remove_attrs=swept_fields,
        extra_attrs=attrs,
    )

    if len(sweep.captures) == 0:
        return

    iq, timestamp = None, None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = zip_offsets(sweep.captures, (-1, 0, 1), fill=None)

    for cap_prev, cap_this, cap_next in offset_captures:
        calls = {}

        if cap_this is not None:
            # extra iteration at the end for the last analysis
            calls['acquire'] = lb.Call(
                radio.acquire, cap_this, next_capture=cap_next, correction=False
            )

        if cap_prev is not None:
            # iq is available after the first iteration
            calls['analyze'] = lb.Call(analyze, iq, timestamp, cap_prev)

        if cap_this is None:
            desc = 'last analysis'
        else:
            # treat swept fields as coordinates/indices
            desc = describe_capture(cap_this, swept_fields)

        with lb.stopwatch(f'{desc} •'):
            ret = lb.concurrently(**calls, flatten=False)

        if 'analyze' in ret:
            # this is what is made available for
            yield ret['analyze']

        if 'acquire' in ret:
            iq, timestamp = ret['acquire']
