from __future__ import annotations
import typing

from frozendict import frozendict

from .radio import RadioDevice, NullRadio
from . import results, structs, util

import channel_analysis

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


def _frozensubset(d: dict | frozendict, keys: list[str]) -> frozendict:
    return frozendict({k: d[k] for k in keys})


def describe_capture(this: structs.RadioCapture, prev: structs.RadioCapture|None = None):
    diffs = {}

    for name in type(this).__struct_fields__:
        if name == 'external':
            continue
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs[name] = value

    this_external = set(this.external.keys())
    prev_external = set() if prev is None else set(prev.external.keys())
    for name in this_external|prev_external:
        value = this.external.get(name, None)

        if prev is None or value != prev.external.get(name, None):
            diffs['external.'+name] = value

    return ', '.join([f'{k}={repr(v)}' for k,v in diffs.items()])


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
    calibration: channel_analysis.DatasetType = None,
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
        'radio_setup': structs.to_builtins(sweep.radio_setup),
        'description': structs.to_builtins(sweep.description),
    }

    analyze = results.ChannelAnalysisWrapper(
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
