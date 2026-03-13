from __future__ import annotations as __

import dataclasses
from math import isfinite
from typing import TYPE_CHECKING

from . import sources, util
from .. import specs


if TYPE_CHECKING:
    import striqt.waveform as sw
    from .typing import Array

else:
    array_api_compat = util.lazy_import('array_api_compat')
    sw = util.lazy_import('striqt.waveform')


# this is experimental, and currently leaves some residual
# time offset in some circumstances
USE_OARESAMPLE = False


def resampling_correction(
    iq: sources.AcquiredIQ, *, axis=1, overwrite_x=False
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

    needs_resample = sources.buffers.needs_resample(iq.resampler, capture)
    needs_filter = isfinite(capture.analysis_bandwidth)

    if not needs_resample:
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
            numtaps=sources.buffers.FILTER_SIZE,
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
    pad = sources.buffers.get_fft_resample_pad(capture, source_spec, iq.analysis)[0]

    return x, pad


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
    pad_in = sources.buffers.get_fft_resample_pad(capture, source_spec, iq.analysis)[0]
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
    oapad = sources.buffers.get_oaresample_pad(capture, source_spec.master_clock_rate)
    lag_pad = sources.buffers.get_trigger_pad_size(source_spec, capture, iq.trigger)
    size_out = round(capture.duration * capture.sample_rate) + round(
        (oapad[1] + lag_pad) * scale
    )
    offset = iq.resampler['nfft_out']

    assert size_out + offset <= x.shape[axis]
    x = sw.axis_slice(x, offset, None, axis=axis)
    assert x.shape[axis] == size_out

    return x, None
