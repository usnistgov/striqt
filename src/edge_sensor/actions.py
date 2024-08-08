from __future__ import annotations
from .radio import RadioBase
from .structs import Sweep, RadioCapture, get_attrs, to_builtins
from .util import zip_offsets
from . import iq_corrections
from channel_analysis.structs import ChannelAnalysis

import labbench as lb
import xarray as xr
import pandas as pd
from channel_analysis import waveform
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Generator

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


@dataclass
class _RadioCaptureAnalyzer:
    """analyze into a dataset"""

    __name__ = 'analysis'

    radio: RadioBase
    analysis_spec: list[ChannelAnalysis]
    remove_attrs: Optional[tuple[str, ...]] = None

    def __call__(self, iq, timestamp, capture) -> xr.Dataset:
        with lb.stopwatch('analyze', logger_level='debug'):
            # for performance, GPU operations are all here in the same thread
            iq = iq_corrections.resampling_correction(iq, capture, self.radio)
            coords = self.get_coords(capture, timestamp=timestamp)

            analysis = waveform.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            ).assign_coords(coords)

        if self.remove_attrs is not None:
            for f in self.remove_attrs:
                del analysis.attrs[f]

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


def sweep_iterator(
    radio: RadioBase, sweep: Sweep, swept_fields: list[str]
) -> Generator[xr.Dataset | None]:
    """sweep through capture acquisition analysis on radio hardware as specified by sweep"""

    analyze = _RadioCaptureAnalyzer(
        radio=radio, analysis_spec=sweep.channel_analysis, remove_attrs=swept_fields
    )

    if len(sweep.captures) == 0:
        return None

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
            # no iq is available yet in the first iteration
            calls['analyze'] = lb.Call(analyze, iq, timestamp, cap_prev)

        if cap_next is None:
            desc = 'last analysis'
        else:
            # treat swept fields as coordinates/indices
            desc = ', '.join([f'{k}={getattr(cap_this, k)}' for k in swept_fields])

        with lb.stopwatch(f'{desc}: '):
            ret = lb.concurrently(**calls, flatten=False)

        if 'analyze' in ret:
            yield ret['analyze']

        if 'acquire' in ret:
            iq, timestamp = ret['acquire']


def sweep_dataset(
    iterator: Generator[xr.Dataset],
    radio: RadioBase,
    sweep: Sweep,
    swept_fields: list[str],
) -> xr.Dataset:
    # step through the captures
    data = [result for result in iterator]

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(swept_fields):
        ds[k].attrs.update(get_attrs(RadioCapture, k))

    ds.attrs['radio_id'] = radio.id
    ds.attrs['radio_setup'] = to_builtins(sweep.radio_setup)
    ds.attrs['description'] = to_builtins(sweep.description)
    ds[TIMESTAMP_NAME].attrs.update(label='Capture start time')

    return ds
