"""implementation of performant acquisition and analysis sequencing for a series of captures"""

from __future__ import annotations
import itertools
import typing
import msgspec

from . import captures, structs, sources, util, xarray_ops
from .peripherals import PeripheralsBase
from .sinks import SinkBase


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
    null_radio_setup = sweep.radio_setup.replace(
        driver=sources.NullSource.__name__,
        resource={},
        _rx_channel_count=radio_cls.rx_channel_count.default,
        calibration=None,
    )

    class WarmupSweep(type(sweep)):
        def get_captures(self):
            # override any capture auto-generating logic
            return structs.Sweep.get_captures(self)

    warmup = WarmupSweep.fromdict(sweep.todict())

    return warmup.replace(captures=captures, radio_setup=null_radio_setup)


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

    c1_compare = c1.replace(**downstream_kws)
    c2_compare = c2.replace(
        # ignore parameters that only affect downstream processing
        # that are validated to have flat response and unity gain
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare


def _build_attrs(sweep: structs.Sweep):
    fields = set(type(sweep).__struct_fields__)
    base_fields = set(structs.Sweep.__struct_fields__)
    new_fields = list(fields - base_fields)
    attr_fields = ['description', 'radio_setup'] + new_fields

    attrs = {}
    for field in attr_fields[::-1]:
        obj = getattr(sweep, field)
        new_attrs = obj.todict()
        attrs.update(new_attrs)

    return attrs


class SweepIterator:
    def __init__(
        self,
        radio: 'sources.SourceBase',
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

        self._peripherals = None
        self._writer = None

        self.setup(sweep)

    def set_peripherals(self, peripherals: PeripheralsBase | None):
        self._peripherals = peripherals

    def set_writer(self, writer: SinkBase | None):
        self._writer = writer

    def setup(self, sweep: structs.Sweep):
        self.sweep: structs.Sweep = sweep.validate()

        self._analyze = xarray_ops.ChannelAnalysisWrapper(
            radio=self.radio,
            sweep=sweep,
            analysis_spec=sweep.channel_analysis,
            extra_attrs=_build_attrs(sweep),
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
            calls = {}

            if capture_prev is None:
                # no pending data in the first iteration
                pass
            else:
                calls['analyze'] = lb.Call(
                    self._analyze,
                    iq,
                    sweep_time=sweep_time,
                    capture=capture_prev,
                    pickled=self._pickled,
                    overwrite_x=not self._reuse_iq,
                    delayed=True,
                )

            if capture_this is None:
                # Nones at the end indicate post-analysis and saves
                pass
            else:
                calls['acquire'] = lb.Call(
                    self._acquire, iq, capture_prev, capture_this, capture_next
                )

            if capture_intake is None:
                # for the first two iterations, there is no data to save
                pass
            else:
                calls['intake'] = lb.Call(
                    self._intake, radio_data=analysis, ext_data=prior_ext_data
                )

            desc = channel_analysis.describe_capture(
                capture_this, capture_prev, index=i, count=count
            )

            with lb.stopwatch(
                f'{desc} •', logger_level='debug' if self._quiet else 'info'
            ):
                ret = util.concurrently_with_fg(calls, flatten=False)

            if 'analyze' in ret:
                # wait until now to do CPU-intensive xarray Dataset packaging
                # in order to leave cycles free to complete the acquisition
                analysis = ret['analyze'].get()

            if 'acquire' in ret:
                iq, capture_prev = ret['acquire']['radio']
                prior_ext_data = this_ext_data
                this_ext_data = ret['acquire'].get('peripherals', {}) or {}

                if not capture_prev.host_resample:
                    assert capture_prev.sample_rate == capture_prev.backend_sample_rate

            if 'intake' in ret:
                yield ret['intake']
            elif self._always_yield:
                yield None

            if sweep_time is None and capture_prev is not None:
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
        calls = {}
        if reuse_this:
            capture_ret = capture_this.replace(
                backend_sample_rate=self.radio.backend_sample_rate(),
                start_time=None,
            )
            calls['radio'] = lb.Call(tuple, (iq_prev, capture_ret))
        else:
            calls['radio'] = lb.Call(
                self.radio.acquire,
                capture_this,
                correction=False,
            )

        if self._peripherals is not None:
            calls['peripherals'] = lb.Call(self._peripherals.acquire, capture_next)

        result = lb.concurrently(**calls, flatten=False)

        if capture_next is not None and not reuse_next:
            self._arm(capture_next)

        return result

    def _arm(self, capture):
        calls = {}

        calls['radio'] = lb.Call(self.radio.arm, capture)
        if self._peripherals is not None:
            calls['peripherals'] = lb.Call(self._peripherals.arm, capture)

        return lb.concurrently(**calls)

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

        if self._writer is not None:
            self._writer.append(radio_data)

        return radio_data


def iter_sweep(
    radio: 'sources.SourceBase',
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


def iter_raw_iq(
    radio: 'sources.SourceBase',
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
        quiet: if True, log at the debug level, and show 'info' level log messages or higher only to the screen

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
