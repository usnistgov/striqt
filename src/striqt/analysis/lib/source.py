from __future__ import annotations as __

import typing

from .. import specs

from . import util

if typing.TYPE_CHECKING:
    import numpy as np

    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayType
else:
    np = util.lazy_import('numpy')
    iqwaveform = util.lazy_import('striqt.waveform')


def filter_iq_capture(
    iq: ArrayType,
    capture: specs.FilteredCapture,
    *,
    axis=0,
    out=None,
) -> ArrayType:
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform
        capture: the capture filter specification structure
        axis: the axis of `x` along which to compute the filter
        out: None, 'shared', or an array object to receive the output data

    Returns:
        the filtered IQ capture
    """

    xp = iqwaveform.util.array_namespace(iq)

    nfft = capture.analysis_filter.nfft
    nfft_out = capture.analysis_filter.nfft_out or nfft

    nfft_out, noverlap, overlap_scale, _ = iqwaveform.fourier._ola_filter_parameters(
        iq.size,
        window=capture.analysis_filter.window,
        nfft_out=nfft_out,
        nfft=nfft,
        extend=True,
    )

    w = iqwaveform.fourier.get_window(
        capture.analysis_filter.window, nfft, fftbins=False, xp=xp
    )
    freqs, _, xstft = iqwaveform.fourier.stft(
        iq,
        fs=capture.sample_rate,
        window=w,
        nperseg=capture.analysis_filter.nfft,
        noverlap=round(capture.analysis_filter.nfft * overlap_scale),
        axis=axis,
        truncate=False,
        out=out,
    )

    freq_res = capture.sample_rate / nfft
    enbw = freq_res * iqwaveform.fourier.equivalent_noise_bandwidth(
        capture.analysis_filter.window, nfft, fftbins=False
    )

    passband = (
        -capture.analysis_bandwidth / 2 + enbw,
        capture.analysis_bandwidth / 2 - enbw,
    )

    if nfft_out != capture.analysis_filter.nfft:
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            nfft_out=capture.analysis_filter.nfft_out or capture.analysis_filter.nfft,
            passband=passband,
            axis=axis,
            out=xstft,
        )

    iqwaveform.fourier.zero_stft_by_freq(freqs, xstft, passband=passband, axis=axis)

    return iqwaveform.fourier.istft(
        xstft,
        iq.shape[axis],
        nfft=nfft_out,
        noverlap=noverlap,
        out=out,
        axis=axis,
    )


def simulated_awgn(
    capture: specs.Capture,
    *,
    power_spectral_density: float = 1,
    xp=np,
    pinned_cuda=False,
    seed=None,
    out=None,
) -> ArrayType:
    # use the slower RandomState for maximum compatibility
    # across array module namespaces
    gen = xp.random.RandomState(seed=seed)
    size = round(capture.duration * capture.sample_rate)

    if out is not None:
        samples = out
    elif pinned_cuda:
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
        samples = None

    x = gen.standard_normal(size=2 * size).astype('float32').view('complex64')

    if samples is None:
        samples = x
    else:
        samples[:] = x

    power = power_spectral_density * capture.sample_rate

    samples *= xp.sqrt(power / 2)

    return samples


def read_tdms(path, analysis_bandwidth: float = float('nan')):
    from nptdms import TdmsFile

    fd = TdmsFile.read(path)

    header_fd, iq_fd = fd.groups()

    size = int(header_fd['total_samples'][0])
    ref_level = header_fd['reference_level_dBm'][0]
    fs = header_fd['IQ_samples_per_second'][0]
    # Support this in the future?
    # fc = header_fd['carrier_frequency'][0]

    scale = 10 ** (float(ref_level) / 20.0) / np.iinfo(np.int16).max
    i, q = iq_fd.channels()
    iq = np.empty((2 * size,), dtype=np.int16)
    iq[::2] = i[:]
    iq[1::2] = q[:]

    frame_size = int(np.rint(10e-3 * fs))
    if iq.shape[0] % frame_size != 0:
        iq = iq[: (iq.shape[0] // frame_size) * frame_size]

    iq = (iq * np.float32(scale)).view('complex64')
    capture = specs.FilteredCapture(
        duration=iq.size / fs, sample_rate=fs, analysis_bandwidth=analysis_bandwidth
    )

    iq = filter_iq_capture(iq, capture)

    return iq, capture
