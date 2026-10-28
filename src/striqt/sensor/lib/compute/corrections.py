from __future__ import annotations as __

import dataclasses
from math import ceil, gcd, isfinite
import os
from typing import cast, TYPE_CHECKING
import warnings

from .. import util
from ... import specs

import striqt.analysis as sa
from striqt.analysis.lib.dataarrays import format_units

if TYPE_CHECKING:
    import striqt.waveform as sw
    from ..typing import Array, ResamplerKws, Unpack

else:
    sw = util.lazy_import('striqt.waveform')

FILTER_SIZE = 4001
FIR_TRANSITION_BW = 250e3
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'

# bypass for the conjugate correction for downconversion based on high-side LO
IGNORE_HIGHSIDE_LO = int(os.environ.get('STRIQT_IGNORE_HIGHSIDE_LO', 0))
if IGNORE_HIGHSIDE_LO:
    warnings.warn(
        'bypassing corrections for highside-LO downconversion '
        '(shell STRIQT_IGNORE_HIGHSIDE_LO=1)'
    )

# experimental: resample and low-pass in one STFT pass instead of _resample and the FIR
USE_OARESAMPLE = int(os.environ.get('STRIQT_USE_OARESAMPLE', 0))
if USE_OARESAMPLE:
    warnings.warn('experimental oaresample is enabled (shell STRIQT_USE_OARESAMPLE=1)')


_TIME_AXIS = 1


def attach_corrections(
    iq: specs.AcquiredIQ,
    *,
    signal_trigger: sa.Trigger | None = None,
    overwrite_x: bool = False,
) -> specs.AcquiredIQ:
    """return a copy of `iq` with the given `signal_trigger` and `overwrite_x`.

    The copy has a `Preprocessor` of its own with every stage empty.

    Args:
        signal_trigger: the trigger that `preprocess` runs for alignment; None
            selects `iq.info.signal_trigger`
        overwrite_x: True to let the resampling stage consume `iq.x` in place

    Raises:
        TypeError: if `iq.capture` is not a `SensorCapture`
    """
    _sensor_capture(iq)
    return dataclasses.replace(
        iq, signal_trigger=signal_trigger, overwrite_x=overwrite_x
    )


class Preprocessor(sa.Preprocessor):
    """the sensor signal path of one acquisition, memoized stage by stage.

    An instance refers to the acquisition weakly and memoizes the stages of that
    acquisition alone. Each stage runs on the first call that needs the stage, and
    `x` of the acquisition must not change after that call.
    """

    def __init__(self, iq: specs.AcquiredIQ) -> None:
        super().__init__(iq)
        self.resampled: tuple[Array, int | None] | None = None
        self.unfiltered: Array | None = None
        self.filtered: Array | None = None
        self.shifts: Array | None = None
        self.outputs: dict[tuple[bool, bool], Array] = {}
        self.aligning: bool = False

    @property
    def iq(self) -> specs.AcquiredIQ:
        """the sensor acquisition that owns this preprocessor.

        Raises:
            RuntimeError: when the acquisition has been garbage collected
        """
        return cast('specs.AcquiredIQ', super().iq)

    def __call__(
        self, *, prefer_filtered: bool = True, prefer_aligned: bool = True
    ) -> Array:
        """return `x` of the acquisition conditioned as the preferences allow.

        The stages, in order: resample to `capture.sample_rate`, conjugate the ports
        in `conjugate`, low-pass to `capture.analysis_bandwidth`, trim to
        `capture.duration`, and shift by the trigger lags.

        Args:
            prefer_filtered: True selects the waveform low-passed to
                `capture.analysis_bandwidth`; the unfiltered waveform is returned
                when `needs_analysis_filter` is false
            prefer_aligned: True selects the waveform shifted by the trigger lags; the
                unaligned waveform is returned when no trigger is set

        Returns:
            shape (port, sample) array of ``round(duration * sample_rate)`` samples
            per port at `capture.sample_rate`, in the namespace of `x`. Repeated calls
            with the same effective conditioning return the same object.

        Raises:
            TypeError: if `capture` is not a `SensorCapture`
            RuntimeError: if called with `prefer_aligned` True from inside the trigger
                measurement, or if the acquisition has been garbage collected
            ValueError: if the acquisition is shorter than the resampler or the
                trigger shifts require
        """
        iq = self.iq
        capture = _sensor_capture(iq)
        filtered = prefer_filtered and needs_analysis_filter(iq)
        aligned = prefer_aligned and _trigger_of(iq) is not None
        key = (filtered, aligned)

        if key in self.outputs:
            return self.outputs[key]
        if aligned and self.aligning:
            raise RuntimeError(
                'the signal_trigger measurement selected prefer_aligned=True while '
                'the alignment was being computed; a trigger must select '
                'prefer_aligned=False'
            )

        base = _filtered_stage(self) if filtered else _unfiltered_stage(self)
        size_out = round(capture.duration * capture.sample_rate)

        if aligned:
            out = _apply_trigger_shifts(base, _trigger_shifts(self), size_out)
        else:
            out = sw.axis_slice(base, 0, size_out, axis=_TIME_AXIS)

        assert out.shape[_TIME_AXIS] == size_out
        self.outputs[key] = out
        return out


def needs_analysis_filter(iq: specs.AcquiredIQ) -> bool:
    """return whether `preprocess` low-passes `iq` to `analysis_bandwidth` with the FIR.

    The filter runs when `analysis_bandwidth` is finite, except on the oaresample
    path, where the resampler applies the low-pass.
    """
    capture = _sensor_capture(iq)
    if not isfinite(capture.analysis_bandwidth):
        return False
    return not (USE_OARESAMPLE and needs_resample(iq.resampler, capture))


def _sensor_capture(iq: specs.AcquiredIQ) -> specs.SensorCapture:
    if not isinstance(iq.capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')
    return iq.capture


def _trigger_of(iq: specs.AcquiredIQ) -> sa.Trigger | None:
    if iq.signal_trigger is not None:
        return iq.signal_trigger
    return iq.info.signal_trigger


def _resampled_stage(pp: Preprocessor) -> tuple[Array, int | None]:
    """return ``(y, lead)``: `x` scaled, resampled, and conjugated, and the lead pad in
    output samples.

    No other stage reads `x` of the acquisition. The stage modifies that `x` in
    place when `overwrite_x` of the acquisition is set.
    """
    if pp.resampled is not None:
        return pp.resampled

    iq = pp.iq
    capture = _sensor_capture(iq)
    max_lag = _get_max_trigger_lag(iq.source_spec, capture, _trigger_of(iq))
    resample_kws = {
        'overwrite_x': iq.overwrite_x,
        'min_overlap': max_lag,
        'axis': _TIME_AXIS,
    }

    with sa.util.stopwatch(
        'resampling filter',
        threshold=capture.duration / 3,
        logger_level=sa.util.PERFORMANCE_INFO,
    ):
        if not needs_resample(iq.resampler, capture):
            y, offs = _scale_only(iq, **resample_kws)
        elif USE_OARESAMPLE:
            y, offs = _oaresample(iq, **resample_kws)
        else:
            y, offs = _resample(iq, **resample_kws)

        if iq.conjugate and not IGNORE_HIGHSIDE_LO:
            # the scale-only path returns the acquisition buffer itself when there
            # is nothing to scale, so conjugate in place only on a fresh array
            y = _apply_conj(
                y, iq.conjugate, overwrite_x=iq.overwrite_x or y is not iq.x
            )

    pp.resampled = (y, offs)
    return pp.resampled


def _unfiltered_stage(pp: Preprocessor) -> Array:
    """return the resampled waveform from the lead offset, with the tail overlap."""
    if pp.unfiltered is None:
        y, offs = _resampled_stage(pp)
        if offs is not None:
            y = sw.axis_slice(y, offs, None, axis=_TIME_AXIS)
        pp.unfiltered = y
    return pp.unfiltered


def _filtered_stage(pp: Preprocessor) -> Array:
    """return the low-passed waveform from the lead offset, with the tail overlap.

    The FIR runs on the full resampled record, and the transient falls in the lead
    overlap. When `needs_analysis_filter` is false, the return value is the
    unfiltered stage object.
    """
    if pp.filtered is not None:
        return pp.filtered

    iq = pp.iq
    if not needs_analysis_filter(iq):
        pp.filtered = _unfiltered_stage(pp)
        return pp.filtered

    capture = _sensor_capture(iq)
    y, offs = _resampled_stage(pp)
    xp = sw.array_namespace(y)
    h = sw.design_fir_lpf(
        bw=capture.analysis_bandwidth,
        fs=capture.sample_rate,
        transition_bw=FIR_TRANSITION_BW,
        numtaps=FILTER_SIZE,
        xp=xp,
    )
    with sa.util.stopwatch(
        'analysis filter',
        threshold=capture.duration / 3,
        logger_level=sa.util.PERFORMANCE_INFO,
    ):
        z = sw.oaconvolve(y, h[xp.newaxis, :], 'same', axes=_TIME_AXIS)
    if offs is not None:
        z = sw.axis_slice(z, offs, None, axis=_TIME_AXIS)
    pp.filtered = z
    return z


def _trigger_shifts(pp: Preprocessor) -> Array:
    """return the per-port shift, in samples, from the trigger of the acquisition.

    `aligning` is set while the trigger measurement runs on the acquisition.
    """
    if pp.shifts is not None:
        return pp.shifts

    iq = pp.iq
    capture = _sensor_capture(iq)
    trigger = _trigger_of(iq)
    assert trigger is not None

    pp.aligning = True
    try:
        lags = trigger(iq, capture)
    finally:
        pp.aligning = False

    xp = sw.array_namespace(lags)
    pp.shifts = xp.rint(lags * capture.sample_rate).astype('int')
    return pp.shifts


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def get_correction_overlaps(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    """return the ``(lead, tail)`` acquisition overlap of `capture` in source samples.

    The lead absorbs the transient of the analysis FIR, which `AcquiredIQ.preprocess`
    applies at `capture.sample_rate` after resampling. The tail absorbs that
    transient and the largest lag of the trigger named by ``setup.signal_trigger``.
    The lead and tail pad the record of `source_record_size` samples to a whole
    number of resampler FFTs.

    Args:
        capture: the capture with the `duration`, `sample_rate`, and
            `analysis_bandwidth` that size the record and the filter
        setup: the source with the `master_clock_rate` that fixes the resampler
            design and the trigger lag
        analysis: the analysis group with the measurement spec of the trigger, or
            None to reserve no tail for alignment

    Returns:
        ``(lead, tail)`` counts of samples at the source rate ``fs_sdr`` of the
        resampler design, each non-negative

    Raises:
        ValueError: if ``setup.signal_trigger`` names a measurement that `analysis`
            does not enable
    """

    from .analyze import get_trigger_from_spec

    trigger = get_trigger_from_spec(setup, analysis)
    max_trigger_lag = _get_max_trigger_lag(setup, capture, trigger)
    return _get_resampler_overlaps(capture, setup, min_overlap=max_trigger_lag)


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def _get_resampler_overlaps(
    capture: specs.SensorCapture, setup: specs.Source, min_overlap: int = 0
) -> tuple[int, int]:
    """return the source-rate overlap that resampling and filtering consume.

    Args:
        min_overlap: tail (in source-rate samples) reserved beyond the filter
            transient for a trigger shift

    Returns:
        ``(lead, tail)`` at the source rate; on the default path, the padded record
        is a whole number of resampler FFTs (`_get_resample_overlap`)
    """

    if USE_OARESAMPLE:
        return _get_oaresample_overlaps(capture, setup.master_clock_rate, min_overlap)
    else:
        return _get_resample_overlap(capture, setup, min_overlap)


def needs_resample(
    analysis_filter: sw.ResamplerDesign, capture: specs.SensorCapture
) -> bool:
    """return whether `capture.host_resample` is set and `analysis_filter` resamples."""
    if not capture.host_resample:
        return False

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


# %% compute resampling parameters based on a given capture and MCR
@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def design_resampler(
    capture: specs.SensorCapture,
    master_clock_rate: float,
    backend_sample_rate: float | None = None,
    **kwargs: Unpack[ResamplerKws],
) -> sw.ResamplerDesign:
    """design a filter specified by the capture for a radio with the specified MCR.

    For the return value, see `striqt.waveform.fourier.design_cola_resampler`
    """
    kwargs.setdefault('bw_lo', 0.25e6)
    kwargs.setdefault('min_oversampling', 1.1)
    kwargs.setdefault('window', RESAMPLE_COLA_WINDOW)
    kwargs.setdefault('min_fft_size', MIN_OARESAMPLE_FFT_SIZE)

    if USE_OARESAMPLE:
        kwargs['min_fft_size'] = MIN_OARESAMPLE_FFT_SIZE
    else:
        kwargs['min_fft_size'] = 256

    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    if (
        capture.analysis_bandwidth != float('inf')
        and capture.analysis_bandwidth > capture.sample_rate
    ):
        raise ValueError(
            f'analysis bandwidth must be smaller than sample rate in {capture}'
        )

    if master_clock_rate is not None:
        mcr = master_clock_rate
    elif capture.backend_sample_rate is not None:
        mcr = capture.backend_sample_rate
    else:
        raise TypeError(
            'must specify source.master_clock_rate or capture.backend_sample_rate'
        )

    if capture.backend_sample_rate is None:
        fs_sdr = backend_sample_rate
    else:
        fs_sdr = capture.backend_sample_rate

    if capture.host_resample:
        # use GPU DSP to resample from integer divisor of the MCR
        design = sw.fourier.design_cola_resampler(
            fs_base=mcr,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=lo_shift,
            fs_sdr=fs_sdr,
            **kwargs,
        )

        if 'window' in kwargs:
            design['window'] = kwargs['window']

        return design

    elif lo_shift:
        raise ValueError('lo_shift requires host_resample=True')
    elif mcr < capture.sample_rate:
        raise ValueError('upsampling requires host_resample=True')
    else:
        # use the SDR firmware to implement the desired sample rate
        return sw.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


def validate_fir_band(capture: specs.SensorCapture) -> None:
    """raise unless the analysis FIR of `capture` has a positive passband below Nyquist.

    The transition band is `FIR_TRANSITION_BW` (Hz) wide and centered on
    ``analysis_bandwidth / 2``. The band must lie strictly inside
    ``(0, sample_rate / 2)``.

    Raises:
        ValueError: if the passband below the transition band is empty, or if the
            transition band reaches ``sample_rate / 2``
    """
    bw = capture.analysis_bandwidth
    fs = capture.sample_rate
    transition = format_units(FIR_TRANSITION_BW, unit='Hz')

    if not bw / 2 - FIR_TRANSITION_BW / 2 > 0:
        raise ValueError(
            f'analysis_bandwidth {format_units(bw, unit="Hz")} leaves no passband '
            f'below the {transition} FIR transition band'
        )
    if not bw / 2 + FIR_TRANSITION_BW / 2 < fs / 2:
        raise ValueError(
            f'analysis_bandwidth {format_units(bw, unit="Hz")} leaves no room for '
            f'the {transition} FIR transition band below the '
            f'{format_units(fs, unit="S/s")} sample_rate'
        )


def validate_oaresample_shift(design: sw.ResamplerDesign) -> None:
    """raise unless `striqt.waveform.oaresample` accepts the ``lo_offset`` of `design`.

    A zero ``lo_offset`` always passes.

    Raises:
        ValueError: if ``lo_offset`` is nonzero while upsampling, is not a multiple
            of the FFT bin spacing ``fs_sdr / nfft`` (Hz) of the resampler, or shifts
            the output passband outside the source bandwidth
    """
    shift = design['lo_offset']
    nfft = design['nfft']
    nfft_out = design['nfft_out']
    bin_size = design['fs_sdr'] / nfft

    if shift == 0:
        return
    if nfft < nfft_out:
        raise ValueError('lo_shift is only supported when downsampling')
    if not sw.isroundmod(shift, bin_size):
        raise ValueError(
            f'the LO offset {format_units(shift, unit="Hz")} is not a multiple of '
            f'the {format_units(bin_size, unit="Hz")} resampler bin'
        )

    edge_low = nfft // 2 - nfft_out // 2 + round(shift / bin_size)
    if edge_low < 0 or edge_low + nfft_out > nfft:
        raise ValueError(
            f'the LO offset {format_units(shift, unit="Hz")} shifts the passband '
            'outside the source bandwidth'
        )


def _apply_trigger_shifts(x: Array, shifts: Array, size_out: int) -> Array:
    if x.shape[1] < shifts.max() + size_out:
        raise ValueError('waveform is too short to align')

    if shifts.shape[0] == 1 or len(set(shifts.tolist())) == 1:
        # fast path: a view when there is only one offset
        return x[:, shifts[0] : shifts[0] + size_out]

    else:
        xp = sw.array_namespace(x)
        out = xp.empty((x.shape[0], size_out), dtype=x.dtype)
        for i in range(x.shape[0]):
            out[i, :] = x[i, shifts[i] : shifts[i] + size_out]
        return out


def _apply_conj(
    x: Array, do_conj: tuple[bool | None, ...], overwrite_x: bool = False
) -> Array:
    assert isinstance(do_conj, tuple)
    xp = sw.array_namespace(x)

    if not any(do_conj):
        return x
    if len(do_conj) != x.shape[-2]:
        raise ValueError(
            'conjugate size mismatch in internal API: this should never happen'
        )
    if not overwrite_x:
        x = x.copy()

    for port, port_conj in enumerate(do_conj):
        if port_conj:
            sub = x[..., port, :]
            xp.conj(sub, out=sub)

    return x


def _scale_only(
    iq: specs.AcquiredIQ, overwrite_x: bool, min_overlap: int, axis: int
) -> tuple[Array, int | None]:
    x = iq.x
    xp = sw.array_namespace(x)
    source_spec = iq.source_spec
    capture = iq.capture

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    vscale = iq.voltage_scale
    if not isinstance(iq.voltage_scale, (int, float)):
        if iq.voltage_scale.ndim == 1:
            vscale = iq.voltage_scale[:, None]
    elif iq.voltage_scale == 1:
        vscale = None

    if vscale is not None:
        x = xp.multiply(x, vscale, out=x if overwrite_x else None)
    overlap = _get_resample_overlap(capture, source_spec, min_overlap)[0]

    return x, overlap


def _resample(
    iq: specs.AcquiredIQ, overwrite_x: bool, min_overlap: int, axis: int
) -> tuple[Array, int | None]:
    """return ``(y, lead)``: `iq.x` resampled to `capture.sample_rate` and the lead pad.

    `y` begins with the lead pad and ends with the resampled tail pad, and `lead` is
    the lead pad in output samples.

    The `lo_offset` of the resampler design is shifted to 0 Hz. `iq.x` is consumed
    in place when `overwrite_x` is set.

    Raises:
        ValueError: if the record does not resample to a whole number of output
            samples, or the LO offset is not a whole number of bins of the record
    """
    x = iq.x
    source_spec = iq.source_spec
    capture = iq.capture
    fs = iq.resampler['fs_sdr']
    lo_offset = iq.resampler['lo_offset']

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    if not sw.isroundmod(x.shape[1] * capture.sample_rate, fs):
        raise ValueError(
            f'{x.shape[1]} samples at {fs} S/s do not resample to a whole number '
            f'of samples at {capture.sample_rate} S/s'
        )
    if not sw.isroundmod(lo_offset * x.shape[axis], fs):
        raise ValueError(
            f'the lo_shift offset {format_units(lo_offset, unit="Hz")} is not a '
            f'whole number of bins of the {x.shape[axis]}-sample record at '
            f'{format_units(fs, unit="S/s")}'
        )
    ny = round(x.shape[1] * capture.sample_rate / fs)
    shift = round(lo_offset * x.shape[axis] / fs)
    padx = _get_resample_overlap(capture, source_spec, min_overlap)[0]
    pady = round(padx * capture.sample_rate / fs)
    scale = 1 if iq.voltage_scale is None else iq.voltage_scale
    y = sw.resample(x, ny, overwrite_x=overwrite_x, axis=axis, scale=scale, shift=shift)

    return y, pady


def _oaresample(
    iq: specs.AcquiredIQ, overwrite_x: bool, min_overlap: int, axis: int
) -> tuple[Array, int | None]:
    """return `iq.x` resampled in one STFT pass and the lead pad in output samples.

    The returned waveform begins with the lead pad and ends with the resampled tail
    pad, at `capture.sample_rate`. The waveform is low-passed to `analysis_bandwidth`
    when finite and shifted by the `lo_offset` of the resampler design. `iq.x` is
    consumed in place when `overwrite_x` is set.

    Raises:
        ValueError: if the acquisition is shorter than the capture plus the pads of
            `_get_oaresample_overlaps` for `min_overlap`
    """
    x = iq.x
    source_spec = iq.source_spec
    capture = iq.capture
    design = iq.resampler
    fs = design['fs_sdr']

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    lead, tail = _get_oaresample_overlaps(
        capture, source_spec.master_clock_rate, min_overlap
    )
    size_in = lead + source_record_size(capture, design) + tail
    if x.shape[axis] < size_in:
        raise ValueError(
            f'{x.shape[axis]} samples at {fs} S/s are fewer than the {size_in} needed '
            f'to resample {capture.duration} s with ({lead}, {tail}) overlaps'
        )

    y = sw.oaresample(
        x,
        up=design['nfft_out'],
        down=design['nfft'],
        fs=fs,
        window=design['window'],
        overwrite_x=overwrite_x,
        axis=axis,
        frequency_shift=design['lo_offset'],
        filter_bandwidth=capture.analysis_bandwidth,
        transition_bandwidth=FIR_TRANSITION_BW,
        scale=1 if iq.voltage_scale is None else iq.voltage_scale,
    )

    return y, round(lead * design['nfft_out'] / design['nfft'])


def _get_filter_overlap(capture: specs.SensorCapture) -> int:
    """return the output-rate sample count of the analysis FIR transient at each end.

    The count is 0 when `analysis_bandwidth` is infinite and no filter runs.
    """
    if isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


def _get_max_trigger_lag(
    setup: specs.Source, capture: specs.SensorCapture, trigger: sa.Trigger | None = None
) -> int:
    if trigger is None:
        return 0

    max_lag = trigger.max_lag(capture)
    lag_pad = ceil(setup.master_clock_rate * max_lag)

    return lag_pad


@sa.util.lru_cache()
def _get_oaresample_overlaps(
    capture: specs.SensorCapture, master_clock_rate: float, min_overlap: int = 0
) -> tuple[int, int]:
    """return the (lead, tail) pads, in source samples, that `_oaresample` needs.

    Each pad covers one STFT overlap: the span at either end of the `sw.oaresample`
    output that fewer than the full count of COLA windows reach. The lead resamples to
    a whole number of output samples. The tail adds `min_overlap` trigger-lag samples
    (in source samples) and is extended until the padded acquisition is a whole
    number of the STFT hops that `sw.oaresample` requires.
    """
    design = design_resampler(capture, master_clock_rate)
    nfft = design['nfft']
    nfft_out = design['nfft_out']

    # only the COLA overlap fraction is needed here; the size argument sizes the
    # padding internal to design_oafilter
    overlap_scale = sw.fourier.design_oafilter(
        nfft, window=design['window'], nfft_out=nfft_out, nfft=nfft, extend=True
    )[2]
    noverlap = round(nfft * overlap_scale)
    hop = nfft - noverlap

    # resampling by nfft_out / nfft maps only multiples of lead_step onto whole output
    # samples
    lead_step = nfft // gcd(nfft, nfft_out)
    lead = sw.util.ceildiv(noverlap, lead_step) * lead_step

    tail = noverlap + min_overlap
    tail += (nfft - (lead + source_record_size(capture, design) + tail)) % hop

    return lead, tail


def source_record_size(capture: specs.SensorCapture, design: sw.ResamplerDesign) -> int:
    """return the source-rate sample count of `capture.duration` under `design`.

    The count is the output sample count ``round(duration * sample_rate)`` scaled by
    the exact ``nfft / nfft_out`` ratio of `design`, in integer arithmetic, with a
    half sample rounded up.
    """
    samples_out = round(capture.duration * capture.sample_rate)
    nfft, nfft_out = design['nfft'], design['nfft_out']
    # integer round-half-up: the float product duration * fs_sdr falls below the half
    # sample when both factors are inexact, and round() would then round down
    return (2 * samples_out * nfft + nfft_out) // (2 * nfft_out)


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def _get_resample_overlap(
    capture: specs.SensorCapture, setup: specs.Source, min_overlap: int = 0
) -> tuple[int, int]:
    """return the (lead, tail) source-rate pad around the record of `capture`.

    The padded record is a whole number of resampler FFTs, and that number is a fast
    FFT length for `setup.array_backend`. The lead is a multiple of the reduced
    denominator of the ``nfft / nfft_out`` ratio of the resampler. The FIR transient
    is counted at `capture.sample_rate` and converted to source-rate samples through
    that ratio.

    Args:
        min_overlap: tail (in source-rate samples) reserved beyond the filter
            transient for a trigger shift

    Returns:
        ``(lead, tail)`` at the source rate, with the lead at least the converted
        transient and the tail at least the converted transient plus `min_overlap`
    """
    design = design_resampler(capture, setup.master_clock_rate)
    nfft = design['nfft']
    nfft_out = design['nfft_out']
    analysis_size = source_record_size(capture, design)

    filter_pad = sw.util.ceildiv(_get_filter_overlap(capture) * nfft, nfft_out)
    lead_min = filter_pad
    tail_min = filter_pad + min_overlap

    block_size = nfft * max(1, sw.util.ceildiv(filter_pad, nfft))
    # a lead that is a multiple of q resamples to a whole number of output samples
    q = nfft // gcd(nfft, nfft_out)

    min_blocks = sw.util.ceildiv(analysis_size + lead_min + tail_min, block_size)
    while True:
        pad_blocks = _get_next_fast_len(min_blocks, array_backend=setup.array_backend)
        pad_end = pad_blocks * block_size - analysis_size
        slack = pad_end - lead_min - tail_min

        lead = (lead_min + slack // 2) // q * q
        if lead < lead_min:
            lead = sw.util.ceildiv(lead_min, q) * q
        tail = pad_end - lead
        if tail >= tail_min:
            return (lead, tail)

        # rounding the lead to a multiple of q left the tail short; one more block
        # adds at least q
        min_blocks = pad_blocks + 1


@sa.util.lru_cache()
def _get_next_fast_len(n, array_backend: specs.types.ArrayBackend) -> int:
    if array_backend == 'cupy':
        import cupyx.scipy.fft as fft  # type: ignore
    elif array_backend == 'numpy':
        import scipy.fft as fft
    else:
        raise TypeError(f'invalid array_backend {array_backend}')

    size = fft.next_fast_len(n)
    assert size is not None, ValueError('failed to determine fft size')
    return size
