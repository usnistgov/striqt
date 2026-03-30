from __future__ import annotations as __

import dataclasses
from math import ceil, isfinite
from typing import overload, TYPE_CHECKING

from . import sources, util
from .. import specs

import striqt.analysis as sa

if TYPE_CHECKING:
    import striqt.waveform as sw
    from .typing import Array, ResamplerKws, Unpack
    import numpy as np

else:
    array_api_compat = util.lazy_import('array_api_compat')
    sw = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


# this is experimental, and currently leaves some residual
# time offset in some circumstances
USE_OARESAMPLE = False
FILTER_SIZE = 4001
MIN_OARESAMPLE_FFT_SIZE = 4 * 4096 - 1
RESAMPLE_COLA_WINDOW = 'hamming'
FILTER_DOMAIN = 'time'


def resampling_correction(
    iq: sources.AcquiredIQ,
    *,
    alias_func: specs.helpers.PathAliasFormatter | None = None,
    analysis: specs.AnalysisGroup | None = None,
    axis=1,
    overwrite_x=False
) -> sources.AcquiredIQ:
    """resample, filter, and apply calibration corrections.

    Args:
        iq: IQ dataclass output by a source
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

    signal_trigger = get_trigger_from_spec(iq.source_spec, analysis)

    iq = dataclasses.replace(
        iq,
        trigger=signal_trigger,
        alias_func=alias_func,
        analysis=analysis
    )

    needs_filter = isfinite(capture.analysis_bandwidth)
    if not needs_resample(iq.resampler, capture):
        x_pre_filter, offs = _scale_only(iq, overwrite_x=overwrite_x, axis=axis)
    elif USE_OARESAMPLE:
        x_pre_filter, offs = _oaresample(iq, overwrite_x=overwrite_x, axis=axis)
        needs_filter = False
    else:
        x_pre_filter, offs = _resample(iq, overwrite_x=overwrite_x, axis=axis)

    # apply the filter here and ensure we're working with a copy if needed
    if needs_filter:
        h = sw.design_fir_lpf(
            bw=capture.analysis_bandwidth,
            fs=capture.sample_rate,
            transition_bw=250e3,
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

    x_pre_align = x_pre_filter[:, :size_out]
    if iq.trigger is not None:
        lags = iq.trigger(x[:, :size_out], capture)
        shifts = xp.rint(lags * capture.sample_rate).astype('int')
        out = _synchronize(x, shifts, size_out)
        x_pre_filter = _synchronize(x_pre_filter, shifts, size_out)
    else:
        out = None

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


def _synchronize(x: Array, shifts: Array, size_out: int) -> Array:
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


def _scale_only(
    iq: sources.AcquiredIQ, overwrite_x: bool, axis: int
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
    overlap = _get_fft_resample_overlap(capture, source_spec, iq.analysis)[0]

    return x, overlap


def _resample(
    iq: sources.AcquiredIQ, overwrite_x: bool, axis: int
) -> tuple[Array, int | None]:
    x = iq.pre_align
    source_spec = iq.source_spec
    capture = iq.capture
    fs = iq.resampler['fs_sdr']

    if not isinstance(capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')

    assert sw.isroundmod(x.shape[1] * capture.sample_rate, fs)
    resample_size_out = round(x.shape[1] * capture.sample_rate / fs)
    pad_in = _get_fft_resample_overlap(capture, source_spec, iq.analysis)[0]
    pad_out = round(pad_in * capture.sample_rate / fs)

    x = sw.resample(
        x,
        resample_size_out,
        overwrite_x=overwrite_x,
        axis=axis,
        scale=1 if iq.voltage_scale is None else iq.voltage_scale,
    )

    return x, pad_out


def _oaresample(
    iq: sources.AcquiredIQ, overwrite_x: bool, axis: int
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
        transition_bandwidth=250e3,
        scale=1 if iq.voltage_scale is None else iq.voltage_scale,
    )
    scale = iq.resampler['nfft_out'] / iq.resampler['nfft']
    oapad = get_oaresample_pad(capture, source_spec.master_clock_rate)
    lag_pad = _get_trigger_holdoff_size(source_spec, capture, iq.trigger)
    size_out = round(capture.duration * capture.sample_rate) + round(
        (oapad[1] + lag_pad) * scale
    )
    offset = iq.resampler['nfft_out']

    assert size_out + offset <= x.shape[axis]
    x = sw.axis_slice(x, offset, None, axis=axis)
    assert x.shape[axis] == size_out

    return x, None


@sa.util.lru_cache()
def get_oaresample_pad(capture: specs.SensorCapture, master_clock_rate: float):
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


def needs_resample(
    analysis_filter: sw.ResamplerDesign, capture: specs.SensorCapture
) -> bool:
    """determine whether host resampling will be needed to filter or resample"""
    if not capture.host_resample:
        return False

    is_resample = analysis_filter['nfft'] != analysis_filter['nfft_out']
    return is_resample and capture.host_resample


#%% Compute overlaps 
@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
def get_overlaps(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    """returns the number of extra overlap acquisition samples to acquire"""

    if USE_OARESAMPLE:
        min_lag_pad = _get_trigger_holdoff_size(setup, capture, analysis)
        oa_pad_low, oa_pad_high = get_oaresample_pad(capture, setup.master_clock_rate)
        return (oa_pad_low, oa_pad_high + min_lag_pad)
    else:
        # this is removed before the FFT, so no need to micromanage its size
        fft_pad = _get_fft_resample_overlap(capture, setup, analysis)

        filter_pad = _get_filter_overlap(capture)
        assert fft_pad[0] > filter_pad and fft_pad[1] > filter_pad

        return (fft_pad[0], fft_pad[1])


def _get_filter_overlap(capture: specs.SensorCapture):
    if isfinite(capture.analysis_bandwidth):
        return FILTER_SIZE // 2 + 1
    else:
        return 0


@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
def _get_fft_resample_overlap(
    capture: specs.SensorCapture,
    setup: specs.Source,
    analysis: specs.AnalysisGroup | None = None,
) -> tuple[int, int]:
    # accommodate the large fft by padding to a fast size that includes at least lag_pad
    min_lag_pad = _get_trigger_holdoff_size(setup, capture, analysis)
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
    min_blocks = block_count + sw.util.ceildiv(min_lag_pad, block_size)

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


#%% compute resampling parameters based on a given capture and MCR
@specs.helpers.convert_capture_arg(specs.SensorCapture)
@sa.util.lru_cache(30000)
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


#%% evaluation of trigger evaluation
def _get_trigger_holdoff_size(
    setup: specs.Source,
    capture: specs.SensorCapture,
    trigger_info: sa.Trigger | specs.AnalysisGroup | None = None,
) -> int:
    if isinstance(trigger_info, specs.AnalysisGroup):
        trigger = get_trigger_from_spec(setup, trigger_info)
    elif isinstance(trigger_info, sa.Trigger):
        trigger = trigger_info
    else:
        return 0

    if trigger is None:
        return 0

    mcr = setup.master_clock_rate
    max_lag = trigger.max_lag(capture)
    lag_pad = ceil(mcr * max_lag)

    return lag_pad


@overload
def get_trigger_from_spec(setup: specs.Source, analysis: None = None) -> None: ...


@overload
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup
) -> sa.Trigger: ...


@sa.util.lru_cache()
def get_trigger_from_spec(
    setup: specs.Source, analysis: specs.AnalysisGroup | None = None
) -> sa.Trigger | None:
    name = get_signal_trigger_name(setup)
    if name is None:
        return None

    if analysis is None and isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger

    if analysis is None:
        meas_name = sa.register.get_signal_trigger_measurement_name(name, sa.registry)
        raise ValueError(
            f'signal_trigger {meas_name!r} requires an analysis specification for {setup.signal_trigger!r}'
        )
    elif isinstance(analysis, specs.AnalysisGroup):
        return sa.Trigger.from_spec(name, analysis, registry=sa.registry)
    elif isinstance(analysis, Analysis):
        return sa.Trigger(setup.signal_trigger, analysis, registry=sa.registry)


@sa.util.lru_cache()
def get_signal_trigger_name(setup: specs.Source) -> str | None:
    if isinstance(setup.signal_trigger, specs.AnalysisGroup):
        analysis = setup.signal_trigger
        meas = {
            name: meas for name, meas in analysis.to_dict().items() if meas is not None
        }
        if len(meas) != 1:
            raise ValueError(
                'specify exactly one trigger for an explicit signal_trigger'
            )
        return list(meas.keys())[0]
    elif isinstance(setup.signal_trigger, str):
        return setup.signal_trigger
    else:
        return None
