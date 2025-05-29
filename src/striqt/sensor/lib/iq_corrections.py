from __future__ import annotations
import typing

from .sources import base, SourceBase, design_capture_resampler
from . import calibration, specs, util
from striqt.analysis.lib.xarray_ops import IQPair

if typing.TYPE_CHECKING:
    import array_api_compat
    import iqwaveform
    import iqwaveform.type_stubs
    import numpy as np
    import xarray as xr

else:
    array_api_compat = util.lazy_import('array_api_compat')
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')


USE_OARESAMPLE = False


def _get_voltage_scale(
    capture: specs.RadioCapture, radio: SourceBase, *, force_calibration=False, xp=np
) -> 'iqwaveform.type_stubs.ArrayLike':
    """compute the scaling factor needed to scale each of N channels of an IQ waveform

    Returns:
        an array of type `xp.ndarray` with shape (N,)
    """
    # to make the best use of the calibration lookup cache, remove extraneous
    # fields in case this is a specialized capture subclass
    bare_capture = capture.replace(start_time=None)

    cal_data = radio.calibration if force_calibration is None else force_calibration
    power_scale = calibration.lookup_power_correction(
        cal_data, bare_capture, radio.base_clock_rate, xp=xp
    )

    transport_dtype = radio._transport_dtype
    if transport_dtype == 'int16':
        dtype_scale = 1.0 / float(np.iinfo(transport_dtype).max)
    else:
        dtype_scale = None

    if power_scale is None and dtype_scale is None:
        return None

    if dtype_scale is None:
        dtype_scale = 1
    if power_scale is None:
        power_scale = 1

    return np.sqrt(power_scale) * dtype_scale


def resampling_correction(
    iq: 'iqwaveform.type_stubs.ArrayType',
    capture: specs.RadioCapture,
    radio: SourceBase,
    force_calibration: typing.Optional['xr.Dataset'] = None,
    *,
    overwrite_x=False,
    axis=1,
) -> IQPair:
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform, as a pinned array
        capture: the capture filter specification structure
        radio: the radio instance that performed the capture
        force_calibration: if specified, this calibration dataset is used rather than loading from file
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ capture
    """

    from striqt.analysis.lib.util import except_on_low_memory

    xp = iqwaveform.util.array_namespace(iq)

    if array_api_compat.is_cupy_array(iq):
        util.configure_cupy()

    scale = _get_voltage_scale(
        capture, radio, force_calibration=force_calibration, xp=xp
    )

    design = design_capture_resampler(radio.base_clock_rate, capture)
    fs = design['fs_sdr']

    except_on_low_memory()

    needs_resample = base.needs_resample(design, capture)

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
        # bail here if host resampling is not needed
        size = round(capture.duration * capture.sample_rate)
        iq = iq[:, :size]
        if scale is not None:
            if scale.ndim == 1:
                scale = scale[:, None]
            iq = xp.multiply(iq, scale, out=iq if overwrite_x else None)
        elif not overwrite_x:
            iq = iq.copy()
        return iq

    except_on_low_memory()

    if radio._aligner is None:
        resample_duration = capture.duration
    else:
        resample_duration = capture.duration + radio._aligner.max_lag(capture)

    resample_size_out = round(resample_duration * capture.sample_rate)

    if USE_OARESAMPLE:
        # this is broken. don't use it yet.
        iq = iqwaveform.fourier.oaresample(
            iq,
            up=design['nfft_out'],
            down=design['nfft'],
            fs=fs,
            window=design['window'],
            overwrite_x=overwrite_x,
            axis=axis,
            frequency_shift=design['lo_offset'],
            filter_bandwidth=capture.analysis_bandwidth,
            transition_bandwidth=250e3,
            scale=1 if scale is None else scale,
        )
        scale = design['nfft_out']/design['nfft']
        oapad = base._get_oaresample_pad(radio.base_clock_rate, capture)
        lag_pad = base._get_aligner_pad_size(
            radio.base_clock_rate, capture, radio._aligner
        )
        # print(oapad, lag_pad, scale)
        size_out = (
            round(capture.duration * capture.sample_rate) + round((oapad[1] + lag_pad) * scale)
        )
        assert size_out <= iq.shape[axis]
        iq = iqwaveform.util.axis_slice(iq, -size_out, iq.shape[axis], axis=axis)
        assert iq.shape[axis] == size_out

    else:
        iq = iqwaveform.fourier.resample(
            iq,
            resample_size_out,
            overwrite_x=overwrite_x,
            axis=axis,
            scale=1 if scale is None else scale,
        )

    size_out = round(capture.duration * capture.sample_rate)

    if radio._aligner is not None:
        align_start = radio._aligner(iq[:, :size_out], capture)
        offset = round(align_start * capture.sample_rate)
        assert iq.shape[1] >= offset + size_out
        iq_aligned = iq[:, offset : offset + size_out]
        iq_unaligned = iq[:, :size_out]
    else:
        iq_aligned = None
        iq_unaligned = iq[:, :size_out]

    assert iq_unaligned.shape[axis] == size_out
    assert iq_aligned is None or iq_aligned.shape[axis] == size_out

    return IQPair(aligned=iq_aligned, unaligned=iq_unaligned)

    # nfft = analysis_filter['nfft']
    # nfft_out, noverlap, overlap_scale, _ = iqwaveform.fourier._ola_filter_parameters(
    #     iq.size,
    #     window=analysis_filter['window'],
    #     nfft_out=analysis_filter.get('nfft_out', nfft),
    #     nfft=nfft,
    #     extend=True,
    # )

    # y = iqwaveform.stft(
    #     iq,
    #     fs=fs,
    #     window=analysis_filter['window'],
    #     nperseg=nfft,
    #     noverlap=round(nfft * overlap_scale),
    #     axis=axis,
    #     truncate=False,
    #     overwrite_x=overwrite_x,
    #     return_axis_arrays=False,
    # )

    # # resample
    # except_on_low_memory()
    # if nfft_out < nfft:
    #     # downsample by trimming frequency
    #     freqs = iqwaveform.fftfreq(nfft, 1 / fs)
    #     freqs, y = iqwaveform.fourier.downsample_stft(
    #         freqs, y, nfft_out=nfft_out, axis=axis, out=y
    #     )
    # elif nfft_out > nfft:
    #     # upsample by zero-padding frequency
    #     pad_left = (nfft_out - nfft) // 2
    #     pad_right = pad_left + (nfft_out - nfft) % 2
    #     y = iqwaveform.util.pad_along_axis(y, [[pad_left, pad_right]], axis=axis + 1)

    # if filter_domain == 'frequency':
    #     lb.util.logger.debug('applying filter in frequency domain')
    #     y = iqwaveform.fourier.stft_fir_lowpass(
    #         y,
    #         sample_rate=capture.sample_rate,
    #         bandwidth=capture.analysis_bandwidth,
    #         transition_bandwidth=500e3,
    #         axis=axis,
    #         out=y
    #     )
    # del iq

    # # reconstruct into a resampled waveform
    # except_on_low_memory()
    # iq = iqwaveform.istft(
    #     y, nfft=nfft_out, noverlap=noverlap, axis=axis, overwrite_x=True
    # )
    # scale = iq.size/size_in

    # start the capture after the padding for transients
    # size_out = round(capture.duration * capture.sample_rate)
    # assert size_out <= iq.shape[axis]
    # iq = iqwaveform.util.axis_slice(iq, -size_out, iq.shape[axis], axis=axis)
    # assert iq.shape[axis] == size_out

    # # apply final scaling
    # if power_scale is not None:
    #     scale *= np.sqrt(power_scale)
    # iq *= scale
