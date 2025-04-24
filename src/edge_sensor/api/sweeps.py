from __future__ import annotations
import copy
import typing
import itertools

import msgspec

from . import captures, util, xarray_ops

from .radio import RadioDevice, NullSource, find_radio_cls_by_name
from . import structs


if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
    import channel_analysis
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')
    channel_analysis = util.lazy_import('channel_analysis')


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


def _convert_captures(
    captures: list[channel_analysis.Capture], type_: type[channel_analysis.Capture]
):
    return [msgspec.convert(c, type_, from_attributes=True) for c in captures]


def design_warmup_sweep(
    sweep: structs.Sweep, skip: tuple[structs.RadioCapture, ...]
) -> structs.Sweep:
    """returns a Sweep object for a NullRadio consisting of capture combinations from
    `sweep`.

    This is meant to trigger expensive python imports and warm up JIT caches
    in order to avoid analysis slowdowns during sweeps.
    """

    # captures that have unique sampling parameters, which are those
    # specified in structs.WaveformCapture
    wcaptures = _convert_captures(sweep.captures, structs.WaveformCapture)
    unique_map = dict(zip(wcaptures, sweep.captures))
    skip_wcaptures = set(_convert_captures(skip, structs.WaveformCapture))
    unique_wcaptures = unique_map.keys() - skip_wcaptures
    captures = [unique_map[c] for c in unique_wcaptures]

    if len(captures) > 1:
        captures = captures[:1]

    radio_cls = find_radio_cls_by_name(sweep.radio_setup.driver)

    null_radio_setup = msgspec.structs.replace(
        sweep.radio_setup,
        driver=NullSource.__name__,
        resource={},
        _rx_channel_count=radio_cls.rx_channel_count.default,
        calibration=None,
    )

    return msgspec.structs.replace(
        sweep,  #
        captures=captures,  #
        radio_setup=null_radio_setup,
    )


def iter_sweep(
    radio: RadioDevice,
    sweep: structs.Sweep,
    calibration: 'xr.Dataset' = None,
    always_yield=False,
    quiet=False,
    pickled=False,
    loop=False,
) -> typing.Generator['xr.Dataset' | bytes | None]:
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

    Returns:
        An iterator of analyzed data
    """

    attrs = {
        # metadata fields
        **structs.struct_to_builtins(sweep.radio_setup),
        **structs.struct_to_builtins(sweep.description),
    }

    analyze = xarray_ops.ChannelAnalysisWrapper(
        radio=radio,
        sweep=sweep,
        analysis_spec=sweep.channel_analysis,
        extra_attrs=attrs,
        correction=True,
    )

    if len(sweep.captures) == 0:
        return

    iq = None
    sweep_time = None
    capture_prev = None

    if loop:
        capture_iter = itertools.cycle(sweep.captures)
        count = float('inf')
    else:
        capture_iter = sweep.captures
        count = len(sweep.captures)

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(capture_iter, (-1, 0, 1), fill=None)

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
            print('analyzing ', capture_prev)
            calls['analyze'] = lb.Call(
                analyze,
                iq,
                sweep_time=sweep_time,
                capture=capture_prev,
                pickled=pickled,
            )

        desc = channel_analysis.describe_capture(
            capture_this, capture_prev, index=i, count=count
        )

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


def iter_raw_iq(
    radio: RadioDevice,
    sweep: structs.Sweep,
    quiet=False,
) -> typing.Generator['xr.Dataset' | bytes | None]:
    """iterate through the sweep and yield the raw IQ vector for each.

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

    Returns:
        An iterator of analyzed data
    """

    if len(sweep.captures) == 0:
        return

    iq = None
    capture_prev = None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(sweep.captures, (0, 1), fill=None)

    for i, (capture_this, capture_next) in enumerate(offset_captures):
        desc = captures.describe_capture(
            capture_this, capture_prev, index=i, count=len(sweep.captures)
        )

        with lb.stopwatch(f'{desc} •', logger_level='debug' if quiet else 'info'):
            # extra iteration at the end for the last analysis
            iq, capture = radio.acquire(
                capture_this,
                next_capture=capture_next,
                correction=False,
            )

        yield iq, capture


def stopiter_as_return(iter):
    try:
        return next(iter)
    except StopIteration:
        return StopIteration


def iter_callbacks(
    sweep_iter: 'xr.Dataset' | bytes | None,
    sweep_spec: structs.Sweep,
    *,
    arm_func: callable[[structs.Capture, structs.RadioSetup], None] | None = None,
    acquire_func: callable[[structs.Capture, structs.RadioSetup], None] | None = None,
    intake_func: callable[['xr.Dataset', structs.Capture], typing.Any] | None = None,
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

    if arm_func is None:

        def arm_func(capture):
            pass
    elif not hasattr(arm_func, '__name__'):
        arm_func.__name__ = 'arm'

    if acquire_func is None:

        def acquire_func(capture):
            pass
    elif not hasattr(acquire_func, '__name__'):
        acquire_func.__name__ = 'acquire'

    if intake_func is None:

        def intake_func(data):
            return data

    elif not hasattr(intake_func, '__name__'):
        intake_func.__name__ = 'save'

    # pairs of (data, capture) from the controller
    data_spec_pairs = itertools.zip_longest(
        sweep_iter, sweep_spec.captures, fillvalue=None
    )

    data = None
    this_capture = sweep_spec.captures[0]
    last_data = None

    while True:
        lb.logger.warning('iter loop start')

        if this_capture is not None:
            arm_func(this_capture, sweep_spec.radio_setup)

        returns = lb.concurrently(
            lb.Call(stopiter_as_return, data_spec_pairs).rename('data'),
            lb.Call(acquire_func, this_capture, sweep_spec.radio_setup).rename(
                'acquire'
            ),
            lb.Call(intake_func, last_data).rename('save'),
            flatten=False,
        )

        yield returns.get('save', None)

        if returns['data'] is StopIteration:
            break
        else:
            (data, this_capture) = returns['data']
            ext_data = returns.get('acquire', {})

        if isinstance(data, xr.Dataset):
            new_dims = {xarray_ops.CAPTURE_DIM: data.capture.size}
            ext_dataarrays = {
                k: xr.DataArray(v).expand_dims(new_dims) for k, v in ext_data.items()
            }
            last_data = data.assign(ext_dataarrays)
