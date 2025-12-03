from __future__ import annotations

import dataclasses
import typing

from . import calibration, captures, specs, util
from .sources import AcquiredIQ, base

if typing.TYPE_CHECKING:
    import numpy as np
    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayLike

else:
    array_api_compat = util.lazy_import('array_api_compat')
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


# this is experimental, and currently leaves some residual
# time offset in some circumstances
USE_OARESAMPLE = False


@util.lru_cache()
def _get_voltage_scale(
    capture_spec: specs.ResampledCapture,
    source_spec: specs.Source,
    *,
    alias_func: captures.PathAliasFormatter | None = None,
    xp=None,
) -> tuple['ArrayLike', 'ArrayLike']:
    """compute the scaling factor needed to scale each of N ports of an IQ waveform

    Returns:
        an array of type `xp.ndarray` with shape (N,)
    """
    xp = xp or np

    if isinstance(source_spec, specs.SoapySource):
        assert isinstance(capture_spec, specs.SoapyCapture)
        power_scale = calibration.lookup_power_correction(
            source_spec.calibration,
            capture_spec,
            source_spec.base_clock_rate,
            alias_func=alias_func,
            xp=xp,
        )
    else:
        power_scale = None

    transport_dtype = source_spec.transport_dtype
    if transport_dtype == 'int16':
        adc_scale = 1.0 / float(np.iinfo(transport_dtype).max)
    else:
        adc_scale = None

    if power_scale is None and adc_scale is None:
        return None, 1

    if adc_scale is None:
        adc_scale = 1
    if power_scale is None:
        power_scale = 1

    return xp.sqrt(power_scale) * adc_scale, adc_scale


def _get_peak_power(iq: AcquiredIQ, xp=None):
    xp = iqwaveform.util.array_namespace(iq.raw)

    iq.capture = typing.cast(specs.ResampledCapture, iq)

    _, prescale = _get_voltage_scale(
        iq.capture, iq.source_spec, alias_func=iq.alias_func, xp=xp
    )

    peak_counts = xp.abs(iq.raw).max(axis=-1)
    unscaled_peak = 20 * xp.log10(peak_counts * prescale) - 3
    return unscaled_peak


def resampling_correction(
    iq_in: AcquiredIQ, overwrite_x=False, axis=1
) -> AcquiredIQ:
    """resample, filter, and correct according to specification in iq_in.

    Args:
        iq: IQ dataclass output by a source
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ waveform
    """

    iq = iq_in.raw
    source_spec = iq_in.source_spec
    xp = iqwaveform.util.array_namespace(iq)

    if not isinstance(iq_in.capture, specs.ResampledCapture):
        raise TypeError('iq.capture must be a capture specification')
    else:
        capture = iq_in.capture

    vscale, _ = _get_voltage_scale(
        capture, source_spec, alias_func=iq_in.alias_func, xp=xp
    )

    extra_data = {}

    if source_spec.uncalibrated_peak_detect:
        extra_data['unscaled_iq_peak'] = _get_peak_power(iq_in)

    if (
        isinstance(source_spec, specs.SoapySource)
        and source_spec.calibration is not None
    ):
        assert isinstance(capture, specs.SoapyCapture)

        extra_data['system_noise'] = calibration.lookup_system_noise_power(
            source_spec.calibration,
            capture,
            source_spec.base_clock_rate,
            alias_func=iq_in.alias_func,
        )

    fs = iq_in.resampler['fs_sdr']

    needs_resample = base.needs_resample(iq_in.resampler, capture)

    # apply the filter here and ensure we're working with a copy if needed
    if not USE_OARESAMPLE and np.isfinite(capture.analysis_bandwidth):
        h = iqwaveform.design_fir_lpf(
            bandwidth=capture.analysis_bandwidth,
            sample_rate=fs,
            transition_bandwidth=250e3,
            numtaps=base.FILTER_SIZE,
            xp=xp,
        )
        pad = base._get_filter_pad(capture)
        iq = iqwaveform.oaconvolve(iq, h[xp.newaxis, :], 'same', axes=axis)
        iq = iqwaveform.util.axis_slice(iq, pad, iq.shape[axis], axis=axis)

        # iq is now a copy, so it can be safely overridden
        overwrite_x = True

    if not needs_resample:
        if vscale is not None:
            if vscale.ndim == 1:
                vscale = vscale[:, None]
            iq = xp.multiply(iq, vscale, out=iq if overwrite_x else None)

    elif USE_OARESAMPLE:
        # this is broken. don't use it yet.
        iq = iqwaveform.fourier.oaresample(
            iq,
            up=iq_in.resampler['nfft_out'],
            down=iq_in.resampler['nfft'],
            fs=fs,
            window=iq_in.resampler['window'],
            overwrite_x=overwrite_x,
            axis=axis,
            frequency_shift=iq_in.resampler['lo_offset'],
            filter_bandwidth=capture.analysis_bandwidth,
            transition_bandwidth=250e3,
            scale=1 if vscale is None else vscale,
        )
        scale = iq_in.resampler['nfft_out'] / iq_in.resampler['nfft']
        oapad = base._get_oaresample_pad(source_spec.base_clock_rate, capture)
        lag_pad = base._get_aligner_pad_size(
            source_spec.base_clock_rate, capture, iq_in.aligner
        )
        size_out = round(capture.duration * capture.sample_rate) + round(
            (oapad[1] + lag_pad) * scale
        )
        offset = iq_in.resampler['nfft_out']

        assert size_out + offset <= iq.shape[axis]
        iq = iqwaveform.util.axis_slice(iq, offset, offset + size_out, axis=axis)
        assert iq.shape[axis] == size_out

    else:
        assert iqwaveform.util.isroundmod(iq.shape[1] * capture.sample_rate, fs)
        resample_size_out = round(iq.shape[1] * capture.sample_rate / fs)
        iq = iqwaveform.fourier.resample(
            iq,
            resample_size_out,
            overwrite_x=overwrite_x,
            axis=axis,
            scale=1 if vscale is None else vscale,
        )

    size_out = round(capture.duration * capture.sample_rate)

    if iq_in.aligner is not None:
        align_start = iq_in.aligner(iq[:, :size_out], capture)
        offset = round(align_start * capture.sample_rate)

        if iq.shape[1] < offset + size_out:
            raise ValueError('waveform is too short to align')

        iq_aligned = iq[:, offset : offset + size_out]
        iq_unaligned = iq[:, :size_out]

    else:
        iq_aligned = None
        iq_unaligned = iq[:, :size_out]

    del iq

    assert iq_unaligned.shape[axis] == size_out
    assert iq_aligned is None or iq_aligned.shape[axis] == size_out

    return dataclasses.replace(
        iq_in,
        aligned=iq_aligned,
        raw=iq_unaligned,
        capture=capture,
        extra_data=iq_in.extra_data | extra_data
    )