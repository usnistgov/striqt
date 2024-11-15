from __future__ import annotations
import typing

from . import structs, util

if typing.TYPE_CHECKING:
    import numpy as np
    import iqwaveform
else:
    np = util.lazy_import('numpy')
    iqwaveform = util.lazy_import('iqwaveform')


def filter_iq_capture(
    iq: 'iqwaveform.util.Array',
    capture: structs.FilteredCapture,
    *,
    axis=0,
    out=None,
) -> 'iqwaveform.util.Array':
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

    nfft = capture.analysis_filter['nfft']

    nfft_out, noverlap, overlap_scale, _ = iqwaveform.fourier._ola_filter_parameters(
        iq.size,
        window=capture.analysis_filter['window'],
        nfft_out=capture.analysis_filter.get('nfft_out', nfft),
        nfft=nfft,
        extend=True,
    )

    w = iqwaveform.fourier._get_window(
        capture.analysis_filter['window'], nfft, fftbins=False, xp=xp
    )
    freqs, _, xstft = iqwaveform.fourier.stft(
        iq,
        fs=capture.sample_rate,
        window=w,
        nperseg=capture.analysis_filter['nfft'],
        noverlap=round(capture.analysis_filter['nfft'] * overlap_scale),
        axis=axis,
        truncate=False,
        out=out,
    )

    freq_res = capture.sample_rate / nfft
    enbw = freq_res * iqwaveform.fourier.equivalent_noise_bandwidth(
        capture.analysis_filter['window'], nfft, fftbins=False
    )

    passband = (
        -capture.analysis_bandwidth / 2 + enbw,
        capture.analysis_bandwidth / 2 - enbw,
    )

    if nfft_out != capture.analysis_filter['nfft']:
        freqs, xstft = iqwaveform.fourier.downsample_stft(
            freqs,
            xstft,
            nfft_out=nfft_out,
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
    capture: structs.Capture,
    *,
    power: float = 1,
    xp=np,
    pinned_cuda=False,
    seed=None,
    dtype='float32',
    out=None,
) -> 'iqwaveform.util.Array':
    try:
        # e.g., numpy
        bitgen = xp.random.PCG64(seed=seed)
    except AttributeError:
        # e.g., cupy
        bitgen = xp.random.MRG32k3a(seed=seed)

    generator = xp.random.Generator(bitgen)
    size = round(capture.duration * capture.sample_rate)

    if isinstance(capture, structs.FilteredCapture):
        size = size + 2 * capture.analysis_filter['nfft']

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
        size=2 * size, dtype=dtype, out=samples.view(dtype)
    )

    if capture.analysis_bandwidth is not None:
        power = power * capture.sample_rate / capture.analysis_bandwidth

    samples *= xp.sqrt(power / 2)

    if isinstance(capture, structs.FilteredCapture):
        return filter_iq_capture(samples, capture)[
            capture.analysis_filter['nfft'] : -capture.analysis_filter['nfft']
        ]
    else:
        return samples


def read_tdms(path, analysis_bandwidth: float = None, dtype='float32'):
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

    iq = (iq * np.dtype(dtype)(scale)).view('complex64')
    capture = structs.FilteredCapture(
        duration=iq.size / fs, sample_rate=fs, analysis_bandwidth=analysis_bandwidth
    )

    iq = filter_iq_capture(iq, capture)

    return iq, capture
