from __future__ import annotations
import typing

import labbench as lb

from . import structs, type_stubs
from .io import load, dump

if typing.TYPE_CHECKING:
    import numpy as np
    import iqwaveform
else:
    np = lb.util.lazy_import('numpy')
    iqwaveform = lb.util.lazy_import('iqwaveform')


def filter_iq_capture(
    iq: type_stubs.ArrayType,
    capture: structs.FilteredCapture,
    *,
    axis=0,
    out=None,
) -> type_stubs.ArrayType:
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform
        capture: the capture filter specification structure
        axis: the axis of `x` along which to compute the filter
        out: None, 'shared', or an array object to receive the output data

    Returns:
        the filtered IQ capture
    """

    xp = iqwaveform.fourier.array_namespace(iq)

    fft_size = capture.analysis_filter['fft_size']

    fft_size_out, noverlap, overlap_scale, _ = (
        iqwaveform.fourier._ola_filter_parameters(
            iq.size,
            window=capture.analysis_filter['window'],
            fft_size_out=capture.analysis_filter.get('fft_size_out', fft_size),
            fft_size=fft_size,
            extend=True,
        )
    )

    w = iqwaveform.fourier._get_window(
        capture.analysis_filter['window'], fft_size, fftbins=False, xp=xp
    )
    freqs, _, xstft = iqwaveform.fourier.stft(
        iq,
        fs=capture.sample_rate,
        window=w,
        nperseg=capture.analysis_filter['fft_size'],
        noverlap=round(capture.analysis_filter['fft_size'] * overlap_scale),
        axis=axis,
        truncate=False,
        out=out,
    )

    enbw = (
        capture.sample_rate
        / fft_size
        * iqwaveform.fourier.equivalent_noise_bandwidth(w, fft_size, fftbins=False)
    )
    passband = (
        -capture.analysis_bandwidth / 2 + enbw,
        capture.analysis_bandwidth / 2 - enbw,
    )

    if fft_size_out != capture.analysis_filter['fft_size']:
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            fft_size_out=fft_size_out,
            passband=passband,
            axis=axis,
            out=xstft,
        )

    iqwaveform.fourier.zero_stft_by_freq(freqs, xstft, passband=passband, axis=axis)

    return iqwaveform.fourier.istft(
        xstft,
        iq.shape[axis],
        fft_size=fft_size_out,
        noverlap=noverlap,
        out=out,
        axis=axis,
    )


def simulated_awgn(
    capture: structs.Capture, *, power: float = 1, xp=np, pinned_cuda=False, out=None
) -> type_stubs.ArrayType:
    try:
        # e.g., numpy
        bitgen = xp.random.PCG64()
    except AttributeError:
        # e.g., cupy
        bitgen = xp.random.MRG32k3a()

    generator = xp.random.Generator(bitgen)
    size = round(capture.duration * capture.sample_rate)

    if pinned_cuda:
        import numba
        import numba.cuda

        samples = numba.cuda.mapped_array(
            (size,),
            dtype=xp.complex64,
            strides=None,
            order='C',
            stream=0,
            portable=False,
            wc=False,
        )

        samples = xp.array(samples, copy=False)
    else:
        samples = xp.empty((size,), dtype=xp.complex64)

    generator.standard_normal(
        size=2 * size, dtype=xp.float32, out=samples.view(xp.float32)
    )

    if capture.analysis_bandwidth is not None:
        power = power * capture.sample_rate / capture.analysis_bandwidth

    samples *= xp.sqrt(power / 2)

    return samples
