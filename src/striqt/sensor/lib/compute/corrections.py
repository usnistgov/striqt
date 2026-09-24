from __future__ import annotations as __

import dataclasses
from math import ceil, isfinite
import os
from typing import TYPE_CHECKING
import warnings

from .. import sources, util
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

# oaresample is experimental, and can leave a residual time offset
USE_OARESAMPLE = int(os.environ.get('STRIQT_USE_OARESAMPLE', 0))
if USE_OARESAMPLE:
    warnings.warn('experimental oaresample is enabled (shell STRIQT_USE_OARESAMPLE=1)')


def correct_iq(
    iq: specs.AcquiredIQ,
    *,
    signal_trigger: sa.Trigger | None = None,
    axis=1,
    overwrite_x=False,
) -> specs.AcquiredIQ:
    """resample, filter, and apply calibration corrections.

    Args:
        iq: IQ dataclass output by a source
        signal_trigger: a `signal.analysis.Trigger` instance to implement a
            waveform start alignment, or `None` to default to iq.info.signal_trigger
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ waveform
    """

    x = iq.pre_align
    capture = iq.capture
    xp = sw.array_namespace(x)

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    if signal_trigger is None:
        # TODO: this will need to be fixed along with the definition in iq.info
        signal_trigger = iq.info.signal_trigger  # ty: ignore

    max_lag = _get_max_trigger_lag(iq.source_spec, capture, signal_trigger)
    resample_kws = {'overwrite_x': overwrite_x, 'min_overlap': max_lag, 'axis': axis}
    needs_filter = isfinite(capture.analysis_bandwidth)
    if not needs_resample(iq.resampler, capture):
        x_pre_filter, offs = _scale_only(iq, **resample_kws)
    elif USE_OARESAMPLE:
        x_pre_filter, offs = _oaresample(iq, **resample_kws)
        needs_filter = False
    else:
        x_pre_filter, offs = _resample(iq, **resample_kws)

    if iq.conjugate and not IGNORE_HIGHSIDE_LO:
        # the scale-only path returns the acquisition buffer itself when there is
        # nothing to scale, so in place is only safe on a fresh array
        x_pre_filter = _apply_conj(
            x_pre_filter,
            iq.conjugate,
            overwrite_x=overwrite_x or x_pre_filter is not iq.pre_align,
        )

    # apply the filter here and ensure we're working with a copy if needed
    if needs_filter:
        h = sw.design_fir_lpf(
            bw=capture.analysis_bandwidth,
            fs=capture.sample_rate,
            transition_bw=FIR_TRANSITION_BW,
            numtaps=FILTER_SIZE,
            xp=xp,
        )
        x = sw.oaconvolve(x_pre_filter, h[xp.newaxis, :], 'same', axes=axis)
    else:
        x = x_pre_filter

    if offs is not None:
        x = sw.axis_slice(x, offs, None, axis=axis)
        x_pre_filter = sw.axis_slice(x_pre_filter, offs, None, axis=axis)

    size_out = round(capture.duration * capture.sample_rate)

    x_pre_align = x[:, :size_out]
    if signal_trigger is None:
        out = None
        x_pre_filter = x_pre_filter[:, :size_out]
    else:
        lags = signal_trigger(x_pre_align[:, :size_out], capture)
        shifts = xp.rint(lags * capture.sample_rate).astype('int')
        out = _apply_trigger_shifts(x, shifts, size_out)
        x_pre_filter = _apply_trigger_shifts(x_pre_filter, shifts, size_out)

    del x

    assert x_pre_align.shape[axis] == size_out
    assert out is None or out.shape[axis] == size_out

    return dataclasses.replace(
        iq,
        aligned=out,
        pre_align=x_pre_align,
        pre_filter=x_pre_filter,
        capture=capture,
        extra_data=iq.extra_data,
    )


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def get_correction_overlaps(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    """returns the number of extra overlap acquisition samples to acquire"""

    from .analyze import get_trigger_from_spec

    trigger = get_trigger_from_spec(setup, analysis)
    max_trigger_lag = _get_max_trigger_lag(setup, capture, trigger)
    return _get_resampler_overlaps(capture, setup, min_overlap=max_trigger_lag)


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def _get_resampler_overlaps(
    capture: specs.SensorCapture, setup: specs.Source, min_overlap: int = 0
) -> tuple[int, int]:
    """returns the number of extra overlap acquisition samples to acquire"""

    if USE_OARESAMPLE:
        oa_pad_low, oa_pad_high = _get_oaresample_overlaps(
            capture, setup.master_clock_rate
        )
        return (oa_pad_low, oa_pad_high + min_overlap)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        fft_pad = _get_resample_overlap(capture, setup, min_overlap)

        filter_pad = _get_filter_overlap(capture)
        assert fft_pad[0] > filter_pad and fft_pad[1] > filter_pad

        return (fft_pad[0], fft_pad[1])


def needs_resample(
    analysis_filter: sw.ResamplerDesign, capture: specs.SensorCapture
) -> bool:
    """determine whether host resampling will be needed to filter or resample"""
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
        # this could probably be set to 1?
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
        # fs_sdr, lo_offset, kws  = iqwaveform.fourier.design_cola_resampler(
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
    """raise what `scipy.signal.firls` rejects inside `design_fir_lpf` for `capture`:
    every band needs a positive width, so the transition band must fit strictly
    inside (0, fs/2)

    Raises:
        ValueError: on `capture.analysis_bandwidth`, in capture vocabulary
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
    """raise what `striqt.waveform.oaresample` rejects in `correct_iq` for the LO
    offset of `design`

    Raises:
        ValueError: on `capture.lo_shift`, in capture vocabulary
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
        # fast path: a simple view, if there is only one offset
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
    x = iq.pre_align
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
    x = iq.pre_align
    source_spec = iq.source_spec
    capture = iq.capture
    fs = iq.resampler['fs_sdr']

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    if not sw.isroundmod(x.shape[1] * capture.sample_rate, fs):
        raise ValueError(
            f'{x.shape[1]} samples at {fs} S/s do not resample to a whole number '
            f'of samples at {capture.sample_rate} S/s'
        )
    ny = round(x.shape[1] * capture.sample_rate / fs)
    padx = _get_resample_overlap(capture, source_spec, min_overlap)[0]
    pady = round(padx * capture.sample_rate / fs)
    scale = 1 if iq.voltage_scale is None else iq.voltage_scale
    y = sw.resample(x, ny, overwrite_x=overwrite_x, axis=axis, scale=scale)

    return y, pady


def _oaresample(
    iq: specs.AcquiredIQ, overwrite_x: bool, min_overlap: int, axis: int
) -> tuple[Array, int | None]:
    x = iq.pre_align
    source_spec = iq.source_spec
    capture = iq.capture
    fs = iq.resampler['fs_sdr']

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    x = sw.oaresample(
        x,
        up=iq.resampler['nfft_out'],
        down=iq.resampler['nfft'],
        fs=fs,
        window=iq.resampler['window'],
        overwrite_x=overwrite_x,
        axis=axis,
        frequency_shift=iq.resampler['lo_offset'],
        filter_bandwidth=capture.analysis_bandwidth,
        transition_bandwidth=FIR_TRANSITION_BW,
        scale=1 if iq.voltage_scale is None else iq.voltage_scale,
    )
    scale = iq.resampler['nfft_out'] / iq.resampler['nfft']
    oapad = _get_oaresample_overlaps(capture, source_spec.master_clock_rate)
    size_out = round(capture.duration * capture.sample_rate) + round(
        (oapad[1] + min_overlap) * scale
    )
    offset = iq.resampler['nfft_out']

    assert size_out + offset <= x.shape[axis]
    x = sw.axis_slice(x, offset, None, axis=axis)
    assert x.shape[axis] == size_out

    return x, None


def _get_filter_overlap(capture: specs.SensorCapture):
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
def _get_oaresample_overlaps(capture: specs.SensorCapture, master_clock_rate: float):
    resampler_design = design_resampler(capture, master_clock_rate)

    nfft = resampler_design['nfft']
    nfft_out = resampler_design.get('nfft_out', nfft)

    samples_out = round(capture.duration * capture.sample_rate)
    min_samples_in = ceil(samples_out * nfft / resampler_design['nfft_out'])

    # round up to an integral number of FFT windows
    samples_in = ceil(min_samples_in / nfft) * nfft + nfft

    noverlap_out = sw.fourier.design_oafilter(
        samples_in,
        window=resampler_design['window'],
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )[1]

    noverlap = ceil(noverlap_out * nfft / nfft_out)

    return (samples_in - min_samples_in) + noverlap + nfft // 2, noverlap


@sa.specs.helpers.lru_cache_on_converted(specs.SensorCapture, maxsize=30000)
def _get_resample_overlap(
    capture: specs.SensorCapture, setup: specs.Source, min_overlap: int = 0
) -> tuple[int, int]:
    # accommodate the large fft by padding to a fast size that includes at least lag_pad
    design = design_resampler(capture, setup.master_clock_rate)
    analysis_size = round(capture.duration * design['fs_sdr'])

    # treat the block size as the minimum number of samples needed for the resampler
    # output to have an integral number of samples
    if isfinite(capture.analysis_bandwidth):
        filter_pad = _get_filter_overlap(capture)
        min_filter_blocks = sw.util.ceildiv(design['nfft'], filter_pad)
        block_size = design['nfft'] * min_filter_blocks
    else:
        block_size = design['nfft']
    block_count = analysis_size // block_size
    min_blocks = block_count + sw.util.ceildiv(min_overlap, block_size)

    # since design_capture_resampler gives us a nice fft size
    # for block_size, then if we make sure pad_blocks is also a nice fft size,
    # then the product (pad_blocks * block_size) will also be a product of small
    # primes
    pad_blocks = _get_next_fast_len(min_blocks + 1, array_backend=setup.array_backend)
    pad_end = pad_blocks * block_size - analysis_size
    assert pad_end % 2 == 0

    return (pad_end // 2, pad_end // 2)


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
