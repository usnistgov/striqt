from __future__ import annotations as __


import dataclasses
import typing

from . import sources, util
from .. import specs


if typing.TYPE_CHECKING:
    import numpy as np
    import striqt.waveform as sw
    from striqt.waveform._typing import ArrayType

else:
    array_api_compat = util.lazy_import('array_api_compat')
    sw = util.lazy_import('striqt.waveform')
    np = util.lazy_import('numpy')


# this is experimental, and currently leaves some residual
# time offset in some circumstances
USE_OARESAMPLE = False


def synchronize(x: ArrayType, shifts: ArrayType, size_out: int) -> ArrayType:
    if x.shape[1] < shifts.max() + size_out:
        raise ValueError('waveform is too short to align')

    if shifts.shape[0] == 1 or len(set(shifts.tolist())) == 1:
        # fast path: a simple view, if there is only one offset
        return x[:, shifts[0] : shifts[0] + size_out]

    else:
        xp = sw.util.array_namespace(x)
        out = xp.empty((x.shape[0], size_out), dtype=x.dtype)
        for i in range(x.shape[0]):
            out[i, :] = x[i, shifts[i] : shifts[i] + size_out]
        return out


def resampling_correction(iq: sources.base.AcquiredIQ, overwrite_x=False, axis=1) -> sources.base.AcquiredIQ:
    """resample, filter, and apply calibration corrections.

    Args:
        iq: IQ dataclass output by a source
        axis: the axis of `x` along which to compute the filter
        overwrite_x: if True, modify the contents of IQ in-place; otherwise, a copy will be returned

    Returns:
        the filtered IQ waveform
    """

    x = iq.pre_align
    source_spec = iq.source_spec
    xp = sw.util.array_namespace(x)

    if not isinstance(iq.capture, specs.SensorCapture):
        raise TypeError('iq.capture must be a capture specification')
    else:
        capture = iq.capture

    fs = iq.resampler['fs_sdr']
    needs_resample = sources.base.needs_resample(iq.resampler, capture)
    needs_filter = np.isfinite(capture.analysis_bandwidth)
    vscale = iq.voltage_scale

    if not needs_resample:
        if not isinstance(vscale, (int, float)):
            if vscale.ndim == 1:
                vscale = vscale[:, None]
        elif vscale == 1:
            vscale = None

        if vscale is not None:
            x = xp.multiply(x, vscale, out=x if overwrite_x else None)
        offset_out = sources.base.get_fft_resample_pad(source_spec, capture, iq.analysis)[0]

    elif USE_OARESAMPLE:
        # this is broken. don't use it yet.
        x = sw.fourier.oaresample(
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
            scale=1 if vscale is None else vscale,
        )
        scale = iq.resampler['nfft_out'] / iq.resampler['nfft']
        oapad = sources.base._get_oaresample_pad(source_spec.master_clock_rate, capture)
        lag_pad = sources.base._get_trigger_pad_size(source_spec, capture, iq.trigger)
        size_out = round(capture.duration * capture.sample_rate) + round(
            (oapad[1] + lag_pad) * scale
        )
        offset = iq.resampler['nfft_out']

        assert size_out + offset <= x.shape[axis]
        x = sw.util.axis_slice(x, offset, offset + size_out, axis=axis)
        assert x.shape[axis] == size_out
        needs_filter = False
        offset_out = None

    else:
        assert sw.util.isroundmod(x.shape[1] * capture.sample_rate, fs)
        resample_size_out = round(x.shape[1] * capture.sample_rate / fs)
        offset_in = sources.base.get_fft_resample_pad(source_spec, capture, iq.analysis)[0]
        offset_out = round(offset_in * capture.sample_rate / fs)

        x = sw.fourier.resample(
            x,
            resample_size_out,
            overwrite_x=overwrite_x,
            axis=axis,
            scale=1 if vscale is None else vscale,
        )

    x_pre_filter = x

    # apply the filter here and ensure we're working with a copy if needed
    if needs_filter:
        h = sw.design_fir_lpf(
            bandwidth=capture.analysis_bandwidth,
            sample_rate=capture.sample_rate,
            transition_bandwidth=250e3,
            numtaps=sources.base.FILTER_SIZE,
            xp=xp,
        )
        x = sw.oaconvolve(x, h[xp.newaxis, :], 'same', axes=axis)

    if offset_out is not None:
        x = sw.util.axis_slice(x, offset_out, x.shape[axis], axis=axis)
        x_pre_filter = sw.util.axis_slice(x_pre_filter, offset_out, x_pre_filter.shape[axis], axis=axis)

    size_out = round(capture.duration * capture.sample_rate)

    x_pre_align = x_pre_filter[:, :size_out]
    if iq.trigger is not None:
        lags = iq.trigger(x[:, :size_out], capture)
        shifts = xp.rint(lags * capture.sample_rate).astype('int')
        out = synchronize(x, shifts, size_out)
        x_pre_filter = synchronize(x_pre_filter, shifts, size_out)
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
