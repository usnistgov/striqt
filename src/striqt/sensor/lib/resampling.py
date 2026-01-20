from __future__ import annotations as __


__all__ = ['resampling_correction']


import dataclasses
from numbers import Number
import typing

from .. import specs

from . import calibration, util
from .sources import AcquiredIQ, _base
from striqt.analysis.specs import AnalysisGroup, Analysis

if typing.TYPE_CHECKING:
    import numpy as np
    import striqt.waveform as waveform
    from striqt.waveform._typing import ArrayType

else:
    array_api_compat = util.lazy_import('array_api_compat')
    waveform = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


# this is experimental, and currently leaves some residual
# time offset in some circumstances
USE_OARESAMPLE = False


def resampling_correction(iq_in: AcquiredIQ, overwrite_x=False, axis=1) -> AcquiredIQ:
    """resample, filter, and apply calibration corrections.

    Args:
        iq: IQ dataclass output by a source
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ waveform
    """

    iq = iq_in.raw
    source_spec = iq_in.source_spec
    xp = waveform.util.array_namespace(iq)

    if not isinstance(iq_in.capture, specs.ResampledCapture):
        raise TypeError('iq.capture must be a capture specification')
    else:
        capture = iq_in.capture

    fs = iq_in.resampler['fs_sdr']

    needs_resample = _base.needs_resample(iq_in.resampler, capture)

    vscale = iq_in.voltage_scale
    if not needs_resample:
        if not isinstance(vscale, (int, float)):
            if vscale.ndim == 1:
                vscale = vscale[:, None]
        elif vscale == 1:
            vscale = None

        if vscale is not None:
            iq = xp.multiply(iq, vscale, out=iq if overwrite_x else None)

    elif USE_OARESAMPLE:
        # this is broken. don't use it yet.
        iq = waveform.fourier.oaresample(
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
        oapad = _base._get_oaresample_pad(source_spec.master_clock_rate, capture)
        lag_pad = _base._get_trigger_pad_size(source_spec, capture, iq_in.trigger)
        size_out = round(capture.duration * capture.sample_rate) + round(
            (oapad[1] + lag_pad) * scale
        )
        offset = iq_in.resampler['nfft_out']

        assert size_out + offset <= iq.shape[axis]
        iq = waveform.util.axis_slice(iq, offset, offset + size_out, axis=axis)
        assert iq.shape[axis] == size_out

    else:
        assert waveform.util.isroundmod(iq.shape[1] * capture.sample_rate, fs)
        resample_size_out = round(iq.shape[1] * capture.sample_rate / fs)
        offset_in = _base.get_fft_resample_pad(source_spec, capture, iq_in.analysis)[0]
        offset_out = round(offset_in * capture.sample_rate / fs)

        iq = waveform.fourier.resample(
            iq,
            resample_size_out,
            overwrite_x=overwrite_x,
            axis=axis,
            scale=1 if vscale is None else vscale,
        )

        # apply the filter here and ensure we're working with a copy if needed
        if np.isfinite(capture.analysis_bandwidth):
            h = waveform.design_fir_lpf(
                bandwidth=capture.analysis_bandwidth,
                sample_rate=capture.sample_rate,
                transition_bandwidth=250e3,
                numtaps=_base.FILTER_SIZE,
                xp=xp,
            )
            # pad = _base._get_filter_pad(capture)
            iq = waveform.oaconvolve(iq, h[xp.newaxis, :], 'same', axes=axis)

            # offset_out = offset_out + pad
            # iq = iqwaveform.util.axis_slice(iq, pad, iq.shape[axis], axis=axis)

            # the freshly allocated iq can be safely overridden
            overwrite_x = True

        iq = waveform.util.axis_slice(iq, offset_out, iq.shape[axis], axis=axis)

    size_out = round(capture.duration * capture.sample_rate)

    if iq_in.trigger is not None:
        align_start = iq_in.trigger(iq[:, :size_out], capture)
        offset = round(float(align_start) * capture.sample_rate)

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
        extra_data=iq_in.extra_data,
    )
