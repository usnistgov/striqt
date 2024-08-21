from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Generator, Any
import typing

import labbench as lb
from frozendict import frozendict

from .radio import RadioDevice, NullRadio
from .structs import Sweep, RadioCapture, get_attrs, to_builtins, describe_capture
from .util import zip_offsets
from . import iq_corrections, structs

from channel_analysis.structs import ChannelAnalysis
from channel_analysis import waveform, type_stubs

if typing.TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
else:
    pd = lb.util.lazy_import('pandas')
    xr = lb.util.lazy_import('xarray')


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

    def __call__(
        self, iq: type_stubs.ArrayType, timestamp, capture: RadioCapture
    ) -> xr.Dataset:
        """analyze iq from a capture and package it into a dataset"""

        with lb.stopwatch('analyze', logger_level='debug'):
            # for performance, GPU operations are all here in the same thread
            iq = iq_corrections.resampling_correction(
                iq, capture, self.radio, force_calibration=self.calibration
            )
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


def _frozensubset(d: dict|frozendict, keys: list[str]) -> frozendict:
    return frozendict({k: d[k] for k in keys})


def design_warmup_sweep(
    sweep: structs.Sweep, skip: tuple[structs.RadioCapture, ...]
) -> structs.Sweep:
    """returns a Sweep object for a NullRadio consisting of capture combinations from
    `sweep` with unique combinations of GPU analysis topologies.

    This is meant to be run with fake data to warm up GPU operations and avoid
    analysis slowdowns during sweeps.
    """

    FIELDS = [
        'duration',
        'sample_rate',
        'analysis_bandwidth',
        'lo_shift',
        'gpu_resample',
    ]

    sweep_map = structs.to_builtins(sweep)
    capture_maps = structs.to_builtins(sweep_map['captures'])
    skip = {_frozensubset(structs.to_builtins(s), FIELDS) for s in skip}

    sweep_map['radio_setup']['driver'] = NullRadio.__name__
    sweep_map['radio_setup']['resource'] = 'empty'

    # the set of unique combinations. frozendict enables comparisons for the set ops.
    warmup_captures = {_frozensubset(d, FIELDS) for d in capture_maps}

    sweep_map['captures'] = warmup_captures - skip

    return structs.convert(sweep_map, type(sweep))


def iter_sweep(
    radio: RadioDevice,
    sweep: Sweep,
    swept_fields: list[str],
    calibration: type_stubs.DatasetType = None,
    always_yield=False,
    quiet=False
) -> Generator[xr.Dataset | None]:
    """iterate through sweep captures on the specified radio, yielding a dataset for each.

    Normally, for performance reasons, the first iteration consists of
    `(capture 1) ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    The `always_yield` argument is provided to allow synchronization of hardware between capture 1 and capture 2:
    `(capture 1) ➔ yield None ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    Added checks are needed to filter out the `None` before recording data.

    Args:
        radio: the device that runs the sweep
        sweep: the specification that configures the sweep
        swept_fields: the list of fields that were explicitly specified in the sweep
        calibration: if specified, the calibration data used to scale the output from full-scale to physical power
        always_yield: if `True`, yield `None` before the second capture

    Returns:
        An iterator of analyzed data
    """

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
        calibration=calibration,
    )

    if len(sweep.captures) == 0:
        return

    iq, timestamp = None, None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = zip_offsets(sweep.captures, (-1, 0, 1), fill=None)

    for cap_prev, cap_this, cap_next in offset_captures:
        lb.logger.info('iterate')
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

        with lb.stopwatch(f'{desc} •', logger_level='debug' if quiet else 'info'):
            ret = lb.concurrently(**calls, flatten=False)

        if 'analyze' in ret:
            # this is what is made available for
            yield ret['analyze']
        elif always_yield:
            yield None

        if 'acquire' in ret:
            iq, timestamp = ret['acquire']
