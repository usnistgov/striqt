from __future__ import annotations
import typing
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import msgspec

from . import captures, structs, sources, util, xarray_ops


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

    radio_cls = sources.find_radio_cls_by_name(sweep.radio_setup.driver)

    # TODO: currently, the base_clock_rate is left as the null radio default.
    # this may cause problems in the future if its default disagrees with another
    # radio
    null_radio_setup = msgspec.structs.replace(
        sweep.radio_setup,
        driver=sources.NullSource.__name__,
        resource={},
        _rx_channel_count=radio_cls.rx_channel_count.default,
        calibration=None,
    )

    return msgspec.structs.replace(
        sweep,  #
        captures=captures,  #
        radio_setup=null_radio_setup,
    )


def _iq_is_reusable(
    c1: structs.RadioCapture | None, c2: structs.RadioCapture, base_clock_rate
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    fsb1 = sources.design_capture_filter(base_clock_rate, c1)[0]
    fsb2 = sources.design_capture_filter(base_clock_rate, c2)[0]

    if fsb1 != fsb2:
        # the realized backend sample rates need to be the same
        return False

    downstream_kws = {
        'host_resample': False,
        'start_time': None,
        'backend_sample_rate': None,
    }

    c1_compare = msgspec.structs.replace(c1, **downstream_kws)

    c2_compare = msgspec.structs.replace(
        c2,
        # ignore parameters that only affect downstream processing
        # that are validated to have flat response and unity gain
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare


class CaptureTransformer:
    def __init__(self, sweep: structs.Sweep):
        self.sweep = sweep

    def __iter__(self) -> typing.Generator[structs.RadioCapture]:
        raise NotImplementedError


class LinearCaptureSequencer(CaptureTransformer):
    def __iter__(self) -> typing.Generator[structs.RadioCapture]:
        yield from self.sweep.captures


class SweepIterator:
    def __init__(
        self,
        radio: 'sources.RadioSource',
        sweep: structs.Sweep,
        *,
        calibration: 'xr.Dataset' = None,
        always_yield=False,
        quiet=False,
        pickled=False,
        loop=False,
        reuse_compatible_iq=False,
    ):
        self.radio = radio

        self._calibration = calibration
        self._always_yield = always_yield
        self._quiet = quiet
        self._pickled = pickled
        self._loop = loop
        self._reuse_iq = reuse_compatible_iq

        self._ext_arm = None
        self._ext_acquire = None
        self._ext_intake = None

        self.setup(sweep)

    def set_callbacks(self, *, arm_func, acquire_func, intake_func):
        self._ext_arm = arm_func
        self._ext_acquire = acquire_func
        self._ext_intake = intake_func

    def setup(self, sweep: structs.Sweep):
        self.sweep: structs.Sweep = sweep

        attrs = {
            # metadata fields
            **structs.struct_to_builtins(sweep.radio_setup),
            **structs.struct_to_builtins(sweep.description),
        }

        self._analyze = xarray_ops.ChannelAnalysisWrapper(
            radio=self.radio,
            sweep=sweep,
            analysis_spec=sweep.channel_analysis,
            extra_attrs=attrs,
            correction=True,
        )
        self._analyze.__name__ = 'analyze'
        self._analyze.__qualname__ = 'analyze'

    def __iter__(self) -> typing.Generator['xr.Dataset' | bytes | None]:
        iq = None
        this_ext_data = {}
        prior_ext_data = {}
        sweep_time = None
        capture_prev = None
        analysis = None

        if self._loop:
            capture_iter = itertools.cycle(self.sweep.captures)
            count = float('inf')
        else:
            capture_iter = self.sweep.captures
            count = len(self.sweep.captures)

        if count == 0:
            return

        # iterate across (previous-1, previous, current, next) captures to support concurrency
        offset_captures = util.zip_offsets(capture_iter, (-2, -1, 0, 1), fill=None)

        for i, (capture_intake, _, capture_this, capture_next) in enumerate(
            offset_captures
        ):
            executor = ThreadPoolExecutor(max_workers=3)
            executor.__enter__()
            futures = {}
            ret = {}

            if capture_this is None:
                # Nones at the end indicate post-analysis and saves
                pass
            else:
                futures['acquire'] = executor.submit(
                    self._acquire, iq, capture_prev, capture_this, capture_next
                )

            if capture_intake is None:
                # for the first two iterations, there is no data to save
                pass
            else:
                futures['intake'] = executor.submit(
                    self._intake, radio_data=analysis, ext_data=prior_ext_data
                )

            desc = channel_analysis.describe_capture(
                capture_this, capture_prev, index=i, count=count
            )

            with lb.stopwatch(
                f'{desc} •', logger_level='debug' if self._quiet else 'info'
            ):
                if capture_prev is None:
                    # no pending data in the first iteration
                    pass
                else:
                    # it is important that CUDA operations happen in the main thread
                    # for performance reasons (cause unknown)
                    ret['analyze'] = self._analyze(
                        iq,
                        sweep_time=sweep_time,
                        capture=capture_prev,
                        pickled=self._pickled,
                        overwrite_x=not self._reuse_iq,
                    )

                future_names = dict(zip(futures.values(), futures.keys()))
                ret.update(
                    {
                        future_names[fut]: fut.result()
                        for fut in as_completed(future_names)
                    }
                )

            if 'analyze' in ret:
                analysis = ret['analyze']

            if 'acquire' in ret:
                iq, capture_prev = ret['acquire']['radio']
                prior_ext_data = this_ext_data
                this_ext_data = ret['acquire'].get('extension', {}) or {}

                if not capture_prev.host_resample:
                    assert capture_prev.sample_rate == capture_prev.backend_sample_rate

            if 'intake' in ret:
                yield ret['intake']
            elif self._always_yield:
                yield None

            if sweep_time is None:
                sweep_time = capture_prev.start_time

    def _acquire(self, iq_prev, capture_prev, capture_this, capture_next):
        if self._reuse_iq:
            reuse_this = _iq_is_reusable(
                capture_prev, capture_this, self.radio.base_clock_rate
            )
            reuse_next = _iq_is_reusable(
                capture_this, capture_next, self.radio.base_clock_rate
            )
        else:
            reuse_this = reuse_next = False

        if capture_prev is None:
            self._arm(capture_this)

        # acquire from the radio and any peripherals
        acquire_calls = {}
        if reuse_this:
            capture_ret = msgspec.structs.replace(
                capture_this,
                backend_sample_rate=self.radio.backend_sample_rate(),
                start_time=None,
            )
            acquire_calls['radio'] = lb.Call(tuple, (iq_prev, capture_ret))
        else:
            acquire_calls['radio'] = lb.Call(
                self.radio.acquire,
                capture_this,
                correction=False,
            )

        if self._ext_acquire is not None:
            acquire_calls['extension'] = lb.Call(
                self._ext_acquire, capture_next, self.sweep.radio_setup
            )

        result = lb.concurrently(**acquire_calls, flatten=False)

        if capture_next is not None and not reuse_next:
            self._arm(capture_next)

        return result

    def _arm(self, capture):
        arm_calls = {}

        arm_calls['radio'] = lb.Call(self.radio.arm, capture)
        if self._ext_arm is not None:
            arm_calls['extension'] = lb.Call(
                self._ext_arm, capture, self.sweep.radio_setup
            )

        return lb.concurrently(**arm_calls)

    def _intake(self, radio_data: 'xr.Dataset', ext_data={}):
        if not isinstance(radio_data, xr.Dataset):
            raise ValueError(
                f'expected xr.Dataset type for data, not {type(radio_data)}'
            )

        if len(ext_data) > 0:
            update_ext_dims = {xarray_ops.CAPTURE_DIM: radio_data.capture.size}
            new_arrays = {
                k: xr.DataArray(v).expand_dims(update_ext_dims)
                for k, v in ext_data.items()
            }
            radio_data = radio_data.assign(new_arrays)

        if self._ext_intake is not None:
            self._ext_intake(radio_data)

        return radio_data


def iter_sweep(
    radio: 'sources.RadioSource',
    sweep: structs.Sweep,
    *,
    calibration: 'xr.Dataset' = None,
    always_yield=False,
    quiet=False,
    pickled=False,
    loop=False,
    reuse_compatible_iq=False,
) -> SweepIterator:
    """iterate through sweep captures on the radio, yielding a dataset for each.

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

    return SweepIterator(**locals())


def iter_callbacks(
    sweep_iter: SweepIterator,
    sweep_spec: structs.Sweep = None,
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

    sweep_iter.set_callbacks(
        arm_func=arm_func, acquire_func=acquire_func, intake_func=intake_func
    )

    return sweep_iter


def iter_raw_iq(
    radio: 'sources.RadioSource',
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


# def stopiter_as_return(iter):
#     try:
#         return next(iter)
#     except StopIteration:
#         return StopIteration

# def iter_callbacks(
#     sweep_iter: 'xr.Dataset' | bytes | None,
#     sweep_spec: structs.Sweep,
#     *,
#     arm_func: callable[[structs.Capture, structs.RadioSetup], None] | None = None,
#     acquire_func: callable[[structs.Capture, structs.RadioSetup], None] | None = None,
#     intake_func: callable[['xr.Dataset', structs.Capture], typing.Any] | None = None,
# ):
#     """trigger callbacks on each sweep iteration.

#     This can add support for external device setup and acquisition. Each callback should be able
#     to accommodate `None` values as sentinels to indicate that no data is available yet (for `save`)
#     or no data being acquired (for `setup` and `acquire`).

#     Args:
#         sweep_iter: a generator returned by `iter_sweep`
#         sweep_spec: the sweep specification for `sweep_iter`
#         setup: function to be called during before the start of each capture
#         acquire: function to be called during the acquisition of each capture
#         save: function to be called after the acquisition and analysis of each capture

#     Returns:
#         Generator
#     """

#     if arm_func is None:

#         def arm_func(capture):
#             pass
#     elif not hasattr(arm_func, '__name__'):
#         arm_func.__name__ = 'arm'

#     if acquire_func is None:

#         def acquire_func(capture):
#             pass
#     elif not hasattr(acquire_func, '__name__'):
#         acquire_func.__name__ = 'acquire'

#     if intake_func is None:

#         def intake_func(data):
#             return data

#     elif not hasattr(intake_func, '__name__'):
#         intake_func.__name__ = 'save'

#     # pairs of (data, capture) from the controller
#     data_spec_pairs = itertools.zip_longest(
#         sweep_iter, sweep_spec.captures, fillvalue=None
#     )

#     data = None
#     this_capture = sweep_spec.captures[0]
#     last_data = None

#     while True:
#         lb.util.logger.warning(f'peripherals arm and acquire for {this_capture}')
#         if this_capture is not None:
#             arm_func(this_capture, sweep_spec.radio_setup)

#         returns = lb.concurrently(
#             lb.Call(stopiter_as_return, data_spec_pairs).rename('data'),
#             lb.Call(acquire_func, this_capture, sweep_spec.radio_setup).rename(
#                 'acquire'
#             ),
#             lb.Call(intake_func, last_data).rename('save'),
#             flatten=False,
#         )

#         yield returns.get('save', None)

#         if returns['data'] is StopIteration:
#             break
#         else:
#             (data, this_capture) = returns['data']
#             ext_data = returns.get('acquire', {})

#         if isinstance(data, xr.Dataset):
#             new_dims = {xarray_ops.CAPTURE_DIM: data.capture.size}
#             ext_dataarrays = {
#                 k: xr.DataArray(v).expand_dims(new_dims) for k, v in ext_data.items()
#             }
#             last_data = data.assign(ext_dataarrays)
