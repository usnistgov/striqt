from __future__ import annotations

import typing

from . import calibration, specs, util
from .sources import (
    AcquiredIQ,
    SourceBase,
    base,
)

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayLike, ArrayType

else:
    array_api_compat = util.lazy_import('array_api_compat')
    iqwaveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')


# this is experimental, and currently leaves some residual
# offset in some circumstances
USE_OARESAMPLE = False


def _get_voltage_scale(
    capture: specs.CaptureSpec,
    radio: SourceBase,
    *,
    force_calibration: 'xr.Dataset|None' = None,
    xp=None,
) -> tuple['ArrayLike', 'ArrayLike']:
    """compute the scaling factor needed to scale each of N ports of an IQ waveform

    Returns:
        an array of type `xp.ndarray` with shape (N,)
    """
    xp = xp or np

    # to make the best use of the calibration lookup cache, remove extraneous
    # fields in case this is a specialized capture subclass
    bare_capture = capture.replace(start_time=None)

    if force_calibration is None:
        cal_data = getattr(radio.__setup__, 'calibration', None)
    else:
        cal_data = force_calibration

    if isinstance(bare_capture, specs.SoapyCaptureSpec):
        power_scale = calibration.lookup_power_correction(
            cal_data, bare_capture, radio.setup_spec.base_clock_rate, xp=xp
        )
    else:
        power_scale = None

    transport_dtype = radio.__setup__.transport_dtype
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


def resampling_correction(
    iq_in: AcquiredIQ,
    capture: specs.CaptureSpec,
    radio: SourceBase,
    force_calibration: typing.Optional['xr.Dataset'] = None,
    *,
    overwrite_x=False,
    axis=1,
) -> AcquiredIQ:
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform, as a pinned array
        capture: the capture filter specification structure
        radio: the radio instance that performed the capture
        force_calibration: if specified, this calibration dataset is used rather than loading from file
        adc_peak: if specified, returns the ADC peak level for overload detection
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ waveform
    """

    iq = iq_in.raw
    xp = iqwaveform.util.array_namespace(iq)

    vscale, prescale = _get_voltage_scale(
        capture, radio, force_calibration=force_calibration, xp=xp
    )

    if radio.__setup__.uncalibrated_peak_detect:
        logger = util.get_logger('analysis')
        peak_counts = xp.abs(iq).max(axis=-1)
        unscaled_peak = 20 * xp.log10(peak_counts * prescale) - 3
        descs = ','.join(f'{p:0.0f}' for p in unscaled_peak)
        logger.info(f'({descs}) dBfs ADC peak')
        extra_data = dict(unscaled_iq_peak=unscaled_peak)
    else:
        extra_data = dict()

    resampler = radio.get_resampler()
    fs = resampler['fs_sdr']

    needs_resample = base.needs_resample(resampler, capture)

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
            up=resampler['nfft_out'],
            down=resampler['nfft'],
            fs=fs,
            window=resampler['window'],
            overwrite_x=overwrite_x,
            axis=axis,
            frequency_shift=resampler['lo_offset'],
            filter_bandwidth=capture.analysis_bandwidth,
            transition_bandwidth=250e3,
            scale=1 if vscale is None else vscale,
        )
        scale = resampler['nfft_out'] / resampler['nfft']
        oapad = base._get_oaresample_pad(radio.setup_spec.base_clock_rate, capture)
        lag_pad = base._get_aligner_pad_size(
            radio.setup_spec.base_clock_rate, capture, radio._aligner
        )
        size_out = round(capture.duration * capture.sample_rate) + round(
            (oapad[1] + lag_pad) * scale
        )
        offset = resampler['nfft_out']

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

    if radio._aligner is not None:
        align_start = radio._aligner(iq[:, :size_out], capture)
        offset = round(align_start * capture.sample_rate)
        assert iq.shape[1] >= offset + size_out, ValueError(
            'waveform is too short to align'
        )

        iq_aligned = iq[:, offset : offset + size_out]
        iq_unaligned = iq[:, :size_out]

    else:
        iq_aligned = None
        iq_unaligned = iq[:, :size_out]

    del iq

    assert iq_unaligned.shape[axis] == size_out
    assert iq_aligned is None or iq_aligned.shape[axis] == size_out

    return AcquiredIQ(
        aligned=iq_aligned,
        raw=iq_unaligned,
        capture=capture,
        info=iq_in.info,
        extra_data=iq_in.extra_data | extra_data,
    )
