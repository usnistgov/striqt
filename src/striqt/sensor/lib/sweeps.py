"""implementation of performant acquisition and analysis sequencing for a series of captures"""

from __future__ import annotations
import contextlib
import itertools
import typing
from collections import Counter

from . import captures, datasets, sources, specs, util
from .calibration import lookup_system_noise_power
from .peripherals import PeripheralsBase
from .sinks import SinkBase
from striqt.analysis.lib.dataarrays import AcquiredIQ

if typing.TYPE_CHECKING:
    import xarray as xr
    import labbench as lb
    from striqt import analysis
else:
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')
    analysis = util.lazy_import('striqt.analysis')


def varied_capture_fields(sweep: specs.Sweep) -> list[str]:
    """generate a list of capture fields with at least 2 values in the specified sweep"""
    unlooped_values = (c.todict().values() for c in sweep.get_captures(False))
    unlooped_counts = [len(Counter(v)) for v in zip(*unlooped_values)]
    fields = sweep.captures[0].todict().keys()
    unlooped_counts = dict(zip(fields, unlooped_counts))
    looped_counts = {loop.field: len(loop.get_points()) for loop in sweep.loops}
    all_counts = {
        field: max(unlooped_counts[field], looped_counts.get(field, 0))
        for field in fields
    }
    return [f for f, c in all_counts.items() if c > 1]


def sweep_touches_gpu(sweep: specs.Sweep):
    """returns True if the sweep would benefit from the GPU"""
    IQ_MEAS_NAME = analysis.measurements.iq_waveform.__name__

    if sweep.radio_setup.calibration is not None:
        return True

    analysis_dict = sweep.analysis.todict()
    if tuple(analysis_dict.keys()) != (IQ_MEAS_NAME,):
        # everything except iq_clipping requires a warmup
        return True

    for capture in sweep.captures:
        if capture.host_resample or capture.analysis_bandwidth is not None:
            return True

    return False


def design_warmup_sweep(
    sweep: specs.Sweep, skip: tuple[specs.RadioCapture, ...]
) -> specs.Sweep:
    """returns a Sweep object for a NullRadio consisting of capture combinations from
    `sweep`.

    This is meant to trigger expensive python imports and warm up JIT caches
    in order to avoid analysis slowdowns during sweeps.
    """

    # captures that have unique sampling parameters, which are those
    # specified in structs.WaveformCapture
    wcaptures = [specs.WaveformCapture.fromspec(c) for c in sweep.captures]
    unique_map = dict(zip(wcaptures, sweep.captures))
    skip_wcaptures = {specs.WaveformCapture.fromspec(c) for c in skip}
    captures = [unique_map[c] for c in unique_map.keys() if c not in skip_wcaptures]

    if len(captures) > 1:
        captures = captures[:1]

    radio_cls = sources.find_radio_cls_by_name(sweep.radio_setup.driver)

    # TODO: currently, the base_clock_rate is left as the null radio default.
    # this may cause problems in the future if its default disagrees with another
    # radio
    null_radio_setup = sweep.radio_setup.replace(
        driver=sources.NullSource.__name__,
        resource={},
        _rx_port_count=radio_cls.rx_port_count.default,
        calibration=None,
    )

    class WarmupSweep(type(sweep)):
        def get_captures(self):
            # override any capture auto-generating logic
            return specs.Sweep.get_captures(self, looped=False)

    warmup = WarmupSweep.fromdict(sweep.todict())

    return warmup.replace(captures=captures, radio_setup=null_radio_setup)


def _iq_is_reusable(
    c1: specs.RadioCapture | None, c2: specs.RadioCapture, base_clock_rate
):
    """return True if c2 is compatible with the raw and uncalibrated IQ acquired for c1"""

    if c1 is None or c2 is None:
        return False

    fsb1 = sources.design_capture_resampler(base_clock_rate, c1)['fs_sdr']
    fsb2 = sources.design_capture_resampler(base_clock_rate, c2)['fs_sdr']

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
        analysis_bandwidth=c1.analysis_bandwidth,
        sample_rate=c1.sample_rate,
        **downstream_kws,
    )

    return c1_compare == c2_compare


def _build_attrs(sweep: specs.Sweep):
    fields = set(type(sweep).__struct_fields__)
    base_fields = set(specs.Sweep.__struct_fields__)
    new_fields = list(fields - base_fields)
    attr_fields = ['radio_setup'] + new_fields

    if isinstance(sweep.description, str):
        attrs = {'description': sweep.description}
    else:
        attrs = {'description': sweep.description.todict()}

    for field in attr_fields[::-1]:
        obj = getattr(sweep, field)
        new_attrs = obj.todict()
        attrs.update(new_attrs)

    return attrs


def _update_sweep_time(
    sweep_time: specs.StartTimeType | None,
    new: specs.RadioCapture,
    old: specs.RadioCapture | None,
) -> specs.StartTimeType:
    if sweep_time is None:
        return new.start_time
    elif old is None:
        return sweep_time
    elif old._sweep_index != new._sweep_index:
        return new.start_time
    else:
        return sweep_time


@contextlib.contextmanager
def log_progress_contexts(index, count):
    """set the log context information for reporting progress"""

    contexts = (
        util.log_capture_context(
            'source', capture_index=index, capture_count=count
        ),
        util.log_capture_context(
            'analysis',
            capture_index=index - 1,
            capture_count=count,
        ),
        util.log_capture_context(
            'sink',
            capture_index=index - 2,
            capture_count=count,
        )
    )

    cm = contextlib.ExitStack()

    with cm:
        try:
            for ctx in contexts:
                cm.enter_context(ctx)
            yield cm
        except:
            cm.close()
            raise


class SweepIterator:
    def __init__(
        self,
        radio: 'sources.SourceBase',
        sweep: specs.Sweep,
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
        self._sink = None

        self.setup(sweep)

    def set_peripherals(self, peripherals: PeripheralsBase | None):
        self._peripherals = peripherals

    def set_writer(self, writer: SinkBase | None):
        self._sink = writer

    def setup(self, sweep: specs.Sweep):
        self.sweep: specs.Sweep = sweep.validate()

        self._analyze = datasets.AnalysisCaller(
            radio=self.radio,
            sweep=sweep,
            analysis_spec=sweep.analysis,
            extra_attrs=_build_attrs(sweep),
            correction=True,
            cache_callback=self.show_cache_info
        )
        self._analyze.__name__ = 'analyze'
        self._analyze.__qualname__ = 'analyze'

    def show_cache_info(self, cache, capture: specs.RadioCapture, result, *_, **__):
        cal = self.sweep.radio_setup.calibration
        if cal is None or 'spectrogram' not in cache.name:
            return
        
        import iqwaveform
        
        spg, attrs = result

        xp = iqwaveform.util.array_namespace(spg)

        # conversion to dB is left for after this function, but display
        # log messages in dB
        peaks = iqwaveform.powtodB(spg.max(axis=tuple(range(1, spg.ndim))))

        noise = lookup_system_noise_power(
            cal,
            capture,
            base_clock_rate=capture.backend_sample_rate,
            B=attrs['noise_bandwidth'],
            xp=xp
        )

        snr = peaks - noise

        snr_desc = ', '.join(f'{p:+0.0f}' for p in snr)
        util.get_logger('analysis').info(
            f'({snr_desc}) dB SNR spectrogram peak'
        )

    def __iter__(self) -> typing.Generator['xr.Dataset' | bytes | None]:
        iq = None
        this_ext_data = {}
        prior_ext_data = {}
        sweep_time = None
        canalyze = None
        result = None

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

        for i, (csink, _, cacquire, cnext) in enumerate(offset_captures):
            calls = {}

            with log_progress_contexts(i, count):
                if iq is None:
                    # no pending data in the first iteration
                    pass
                else:
                    calls['analyze'] = lb.Call(
                        self._analyze,
                        iq,
                        sweep_time=sweep_time,
                        capture=canalyze,
                        pickled=self._pickled,
                        overwrite_x=not self._reuse_iq,
                        delayed=True,
                        block_each=False,
                    )

                if cacquire is None:
                    # this happens at the end during the last post-analysis and intakes
                    pass
                else:
                    calls['acquire'] = lb.Call(
                        self._acquire, iq, canalyze, cacquire, cnext
                    )

                if result is None:
                    # for the first two iterations, there is no data to save
                    pass
                else:
                    assert csink is not None
                    result.set_extra_data(prior_ext_data)
                    calls['intake'] = lb.Call(
                        self._intake,
                        results=result.to_xarray(),
                        capture=csink,
                    )

                ret = util.concurrently_with_fg(calls, flatten=False)

                result = ret.get('analyze', None)

                if 'acquire' in ret:
                    iq = ret['acquire']['radio']
                    sweep_time = _update_sweep_time(sweep_time, iq.capture, canalyze)
                    canalyze = iq.capture
                    prior_ext_data = this_ext_data
                    this_ext_data = ret['acquire'].get('peripherals', {}) or {}
                else:
                    iq = None
                    canalyze = None

                if 'intake' in ret:
                    yield ret['intake']
                elif self._always_yield:
                    yield None

    @util.stopwatch('✓', 'source', logger_level=util.PERFORMANCE_INFO)
    def _acquire(self, iq_prev: AcquiredIQ, capture_prev, capture_this, capture_next):
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
            ret_iq = AcquiredIQ(
                raw=iq_prev.raw, aligned=iq_prev.aligned, capture=capture_ret
            )
            calls['radio'] = lb.Call(lambda: ret_iq)
        else:
            calls['radio'] = lb.Call(
                self.radio.acquire,
                capture_this,
                correction=False,
            )

        if self._peripherals is not None:
            calls['peripherals'] = lb.Call(self._peripherals_acquire, capture_next)

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

    def _peripherals_acquire(self, capture):
        if capture is None:
            return {}

        data = self._peripherals.acquire(capture)

        if data is None:
            data = {}
        else:
            try:
                data = dict(data)
            except TypeError:
                raise TypeError(
                    f'{self._peripherals.acquire!r} must return a '
                    f'dict or None, not {type(data)!r}'
                )

        if self.sweep.radio_setup.calibration is None:
            return data

        system_noise = lookup_system_noise_power(
            self.sweep.radio_setup.calibration, capture, self.radio.base_clock_rate
        )

        return dict(data, sensor_system_noise=system_noise)

    @util.stopwatch('✓', 'sink', threshold=10e-3, logger_level=util.PERFORMANCE_INFO)
    def _intake(
        self,
        results: 'xr.Dataset',
        capture: specs.RadioCapture,
    ) -> 'xr.Dataset' | None:
        if not isinstance(results, xr.Dataset):
            raise ValueError(
                f'expected DelayedAnalysisResult type for data, not {type(results)}'
            )

        if self._sink is None:
            return results
        else:
            self._sink.append(results, capture)


def iter_sweep(
    radio: 'sources.SourceBase',
    sweep: specs.Sweep,
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
    sweep: specs.Sweep,
    quiet=False,
) -> typing.Generator[AcquiredIQ]:
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

    capture_prev = None

    # iterate across (previous, current, next) captures to support concurrency
    offset_captures = util.zip_offsets(sweep.captures, (0, 1), fill=None)

    for i, (capture_this, capture_next) in enumerate(offset_captures):
        desc = captures.describe_capture(
            capture_this, capture_prev, index=i, count=len(sweep.captures)
        )

        with util.stopwatch(
            desc,
            'source',
            logger_level=util.PERFORMANCE_DETAIL if quiet else util.PERFORMANCE_INFO,
        ):
            # extra iteration at the end for the last analysis
            yield radio.acquire(
                capture_this,
                next_capture=capture_next,
                correction=False,
            )
