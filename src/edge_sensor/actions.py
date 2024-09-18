from __future__ import annotations
import dataclasses
import functools
import pickle
import typing

import labbench as lb
from frozendict import frozendict

from .radio import RadioDevice, NullRadio
from . import iq_corrections, structs, util

import channel_analysis
from channel_analysis import type_stubs

if typing.TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
else:
    pd = lb.util.lazy_import('pandas')
    xr = lb.util.lazy_import('xarray')


CAPTURE_DIM = 'capture'
SWEEP_DIM = 'sweep'

CAPTURE_TIMESTAMP_NAME = 'capture_time'
SWEEP_TIMESTAMP_NAME = 'sweep_time'


@functools.lru_cache
def _capture_coord_template(external_fields: frozendict[str, typing.Any]):
    """returns a cached set of xarray coordinate for the given list of swept fields"""

    capture = structs.RadioCapture()
    coords = {}

    for field in structs.RadioCapture.__struct_fields__:
        value = getattr(capture, field)

        coords[field] = xr.Variable(
            (CAPTURE_DIM, SWEEP_DIM), [[value]], fastpath=True
        )

    for field, value in external_fields.items():
        coords[field] = xr.Variable(
            (CAPTURE_DIM, SWEEP_DIM), [[value]], fastpath=True
        )

    coords[CAPTURE_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,), [pd.Timestamp('now')], fastpath=True
    )
    coords[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (SWEEP_DIM,), [pd.Timestamp('now')], fastpath=True
    )

    return xr.Coordinates(coords)


@dataclasses.dataclass
class _RadioCaptureAnalyzer:
    """an IQ data analysis/packaging manager given a radio and desired channel analyses"""

    __name__ = 'analyze'

    radio: RadioDevice
    analysis_spec: list[channel_analysis.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any]|None = None
    calibration: xr.Dataset|None = None

    def __call__(
        self,
        iq: type_stubs.ArrayType,
        capture_time,
        sweep_time,
        capture: structs.RadioCapture,
        pickled=False,
    ) -> xr.Dataset:
        """analyze iq from a capture and package it into a dataset"""

        with lb.stopwatch('analyze', logger_level='debug'):
            # for performance, GPU operations are all here in the same thread
            iq = iq_corrections.resampling_correction(
                iq, capture, self.radio, force_calibration=self.calibration
            )
            coords = self.get_coords(
                capture, capture_time=capture_time, sweep_time=sweep_time
            )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            )

            extra_dims = (CAPTURE_DIM, SWEEP_DIM)
            analysis = analysis.expand_dims(extra_dims).assign_coords(coords)

        if self.extra_attrs is not None:
            analysis.attrs.update(self.extra_attrs)

        analysis[CAPTURE_TIMESTAMP_NAME].attrs.update(label='Capture start time')
        analysis[SWEEP_TIMESTAMP_NAME].attrs.update(label='Sweep start time')

        if pickled:
            return pickle.dumps(analysis)
        else:
            return analysis

    def get_coords(self, capture: structs.RadioCapture, capture_time, sweep_time):
        coords = _capture_coord_template(capture.external).copy(deep=True)

        # for speed, update the values, rather than instantiating new coords from scratch
        for field in capture.__struct_fields__:
            value = getattr(capture, field)
            if isinstance(value, str):
                # to coerce strings as variable-length types later for storage
                coords[field] = coords[field].astype('object')
            coords[field].values[:] = value

        if capture_time is not None:
            coords[CAPTURE_TIMESTAMP_NAME].values[:] = capture_time
            coords[SWEEP_TIMESTAMP_NAME].values[:] = sweep_time

        return coords


def _frozensubset(d: dict | frozendict, keys: list[str]) -> frozendict:
    return frozendict({k: d[k] for k in keys})


def describe_capture(this: structs.RadioCapture, prev: structs.RadioCapture|None = None):
    diffs = {}

    for name in type(this).__struct_fields__:
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs[name] = value

    this_external = set(this.external.keys())
    prev_external = set() if prev is None else set(prev.external.keys())
    for name in this_external|prev_external:
        value = this.external.get(name, None)

        if prev is None or value != prev.external.get(name, None):
            diffs['external.'+name] = value

    return ', '.join([f'{k}={getattr(this, k)}' for k in diffs])


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
    sweep: structs.Sweep,
    calibration: type_stubs.DatasetType = None,
    always_yield=False,
    quiet=False,
    pickled=False,
    close_after=False,
) -> typing.Generator[xr.Dataset | bytes | None]:
    """iterate through sweep captures on the specified radio, yielding a dataset for each.

    Normally, for performance reasons, the first iteration consists of
    `(capture 1) ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    The `always_yield` argument is provided to allow synchronization of hardware between capture 1 and capture 2:
    `(capture 1) ➔ yield None ➔ concurrent(capture 2, analysis 1) ➔ (yield analysis 1)`.
    Added checks are needed to handle `None` before recording data.

    Args:
        radio: the device that runs the sweep
        sweep: the specification that configures the sweep
        calibration: if specified, the calibration data used to scale the output from full-scale to physical power
        always_yield: if `True`, yield `None` before the second capture
        quiet: if True, log at the debug level, and show 'info' level log messages or higher only to the screen
        pickled: if True, yield pickled `bytes` instead of xr.Datasets
        close_after: if True, close the radio after the last capture

    Returns:
        An iterator of analyzed data
    """

    attrs = {
        # metadata fields
        'radio_id': radio.id,
        'radio_setup': structs.to_builtins(sweep.radio_setup),
        'description': structs.to_builtins(sweep.description),
    }

    analyze = _RadioCaptureAnalyzer(
        radio=radio,
        analysis_spec=sweep.channel_analysis,
        extra_attrs=attrs,
        calibration=calibration,
    )

    if len(sweep.captures) == 0:
        return

    iq = None
    capture_time = None
    sweep_time = None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(sweep.captures, (-1, 0, 1), fill=None)

    try:
        for capture_prev, capture_this, capture_next in offset_captures:
            calls = {}

            if capture_this is not None:
                # extra iteration at the end for the last analysis
                calls['acquire'] = lb.Call(
                    radio.acquire,
                    capture_this,
                    next_capture=capture_next,
                    correction=False,
                )

            if capture_prev is not None:
                # iq is only available after the first iteration
                calls['analyze'] = lb.Call(
                    analyze,
                    iq,
                    capture_time=capture_time,
                    sweep_time=sweep_time,
                    capture=capture_prev,
                    pickled=pickled,
                )

            if capture_this is None:
                desc = 'last analysis'
            else:
                # treat swept fields as coordinates/indices
                desc = describe_capture(capture_this, capture_prev)

            with lb.stopwatch(f'{desc} •', logger_level='debug' if quiet else 'info'):
                ret = lb.concurrently(**calls, flatten=False)

            if 'analyze' in ret:
                yield ret['analyze']
            elif always_yield:
                yield None

            if 'acquire' in ret:
                iq, capture_time = ret['acquire']
                if sweep_time is None:
                    sweep_time = capture_time

    finally:
        if close_after:
            radio.close()
