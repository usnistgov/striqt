from __future__ import annotations
import typing
import itertools

from frozendict import frozendict

from . import captures, util

from .radio import RadioDevice, NullSource
from . import structs


if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
    import channel_analysis
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')
    channel_analysis = util.lazy_import('channel_analysis')


def freezefromkeys(d: dict | frozendict, keys: list[str]) -> frozendict:
    return frozendict({k: d[k] for k in keys})


def sweep_touches_gpu(sweep: structs.Sweep):
    """returns True if the sweep would benefit from the GPU"""
    IQ_MEAS_NAME = channel_analysis.measurements.iq_waveform.__name__

    if sweep.radio_setup.calibration is not None:
        return True

    if tuple(sweep.channel_analysis.keys()) != (IQ_MEAS_NAME,):
        # everything except iq_clipping requires a warmup
        return True

    for capture in sweep.captures:
        if capture.host_resample or capture.analysis_bandwidth is not None:
            return True
        
    return False


def design_warmup_sweep(
    sweep: structs.Sweep, skip: tuple[structs.RadioCapture, ...]
) -> structs.Sweep:
    """returns a Sweep object for a NullRadio consisting of capture combinations from
    `sweep` with all unique combinations of data shapes.

    This is meant to be run with fake data to warm up JIT caches and avoid
    analysis slowdowns during sweeps.
    """

    FIELDS = [
        'duration',
        'sample_rate',
        'analysis_bandwidth',
        'lo_shift',
        'host_resample',
    ]


    sweep_map = structs.struct_to_builtins(sweep)
    capture_maps = [
        frozendict(d) for d in structs.struct_to_builtins(sweep_map['captures'])
    ]
    skip = {frozendict(structs.struct_to_builtins(s)) for s in skip}

    sweep_map['radio_setup']['driver'] = NullSource.__name__
    sweep_map['radio_setup']['resource'] = 'empty'

    # key on unique combinations of the desired fields.
    # we are left with only one capture for each combination.
    warmup_mapping = {freezefromkeys(d, FIELDS): d for d in capture_maps}

    sweep_map['captures'] = set(warmup_mapping.values()) - set(skip)

    return structs.builtins_to_struct(sweep_map, type(sweep))


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
        **structs.struct_to_builtins(sweep.radio_setup),
        **structs.struct_to_builtins(sweep.description),
    }

    analyze = captures.ChannelAnalysisWrapper(
        radio=radio,
        sweep=sweep,
        analysis_spec=sweep.channel_analysis,
        extra_attrs=attrs,
        calibration=calibration,
    )

    if len(sweep.captures) == 0:
        return

    iq = None
    sweep_time = None
    capture_prev = None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(sweep.captures, (-1, 0, 1), fill=None)

    try:
        for i, (_, capture_this, capture_next) in enumerate(offset_captures):
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
                    sweep_time=sweep_time,
                    capture=capture_prev,
                    pickled=pickled,
                )

            desc = captures.describe_capture(capture_this, capture_prev, index=i, count=len(sweep.captures))

            with lb.stopwatch(f'{desc} •', logger_level='debug' if quiet else 'info'):
                ret = lb.concurrently(**calls, flatten=False)

            if 'analyze' in ret:
                yield ret['analyze']
            elif always_yield:
                yield None

            if 'acquire' in ret:
                iq, capture_prev = ret['acquire']
                if sweep_time is None:
                    sweep_time = capture_prev.start_time

    finally:
        if close_after:
            radio.close()


def iter_raw_iq_captures(
    radio: RadioDevice,
    sweep: structs.Sweep,
    quiet=False,
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

    if len(sweep.captures) == 0:
        return

    iq = None
    sweep_time = None
    capture_prev = None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(sweep.captures, (0, 1), fill=None)

    try:
        for i, (capture_this, capture_next) in enumerate(offset_captures):
            desc = captures.describe_capture(capture_this, capture_prev, index=i, count=len(sweep.captures))

            with lb.stopwatch(f'{desc} •', logger_level='debug' if quiet else 'info'):
                # extra iteration at the end for the last analysis
                iq = radio.acquire(
                    capture_this,
                    next_capture=capture_next,
                    correction=False,
                )

            yield iq

            if sweep_time is None:
                sweep_time = capture_prev.start_time

    finally:
        if close_after:
            radio.close()


def stopiter_as_return(iter):
    try:
        return next(iter)
    except StopIteration:
        return StopIteration


def iter_callbacks(
    sweep_iter: xr.Dataset | bytes | None,
    sweep_spec: structs.Sweep,
    *,
    setup: callable[[structs.Capture], None] | None = None,
    acquire: callable[[structs.Capture], None] | None = None,
    save: callable[[channel_analysis.DatasetType, structs.Capture], typing.Any]
    | None = None,
):
    """trigger callbacks on each sweep iteration.

    This can add support for external device setup and acquisition. Each callback should be able
    to accommodate `None` values as sentinels to indicate that no data is available yet (for `save`)
    or no data being acquired (for `setup` and `acquire`).

    Args:
        sweep_iter: a generator returned by `iter_sweep`
        sweep_spec: the sweep specification for `sweep_iter`
        setup: function to be called during before the start of each capture
        acquire: function to be called during the acquisition of each capture
        save: function to be called after the acquisition and analysis of each capture

    Returns:
        Generator
    """
    if setup is None:

        def setup(capture):
            pass
    elif not hasattr(setup, '__name__'):
        setup.__name__ = 'setup'

    if acquire is None:
        def acquire(capture):
            pass
    elif not hasattr(acquire, '__name__'):
        acquire.__name__ = 'acquire'

    if save is None:

        def save(data):
            return data
    elif not hasattr(save, '__name__'):
        save.__name__ = 'save'

    # pairs of (data, capture) from the controller
    data_spec_pairs = itertools.zip_longest(
        sweep_iter, sweep_spec.captures, fillvalue=None
    )

    data = None
    this_capture = sweep_spec.captures[0]
    last_data = None

    while True:
        if this_capture is not None:
            setup(this_capture)

        returns = lb.concurrently(
            lb.Call(stopiter_as_return, data_spec_pairs).rename('data'),
            lb.Call(acquire, this_capture).rename('acquire'),
            lb.Call(save, last_data).rename('save'),
            flatten=False,
        )

        yield returns.get('save', None)

        if returns['data'] is StopIteration:
            break
        else:
            (data, this_capture) = returns['data']
            ext_data = returns['acquire']

        if data is not None:
            ext_dataarrays = {
                k: xr.DataArray(v).expand_dims(captures.CAPTURE_DIM)
                for k, v in ext_data.items()
            }
            last_data = data.assign(ext_dataarrays)
