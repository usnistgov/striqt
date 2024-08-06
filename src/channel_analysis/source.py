from __future__ import annotations
import numpy as np
from iqwaveform import fourier
from functools import wraps
from . import structs, waveform
import xarray as xr
from pathlib import Path
import zarr
import numcodecs


def dump(path: str | Path, data: xr.DataArray | xr.Dataset, mode='a'):
    """serialize a dataset into a zarr directory structure"""
    if hasattr(data, waveform.IQ_WAVEFORM_INDEX_NAME):
        if 'sample_rate' in data.attrs:
            # sample rate is metadata
            sample_rate = data.attrs['sample_rate']
        else:
            # sample rate is a variate
            sample_rate = data.sample_rate.values.flatten()[0]

        chunks = {waveform.IQ_WAVEFORM_INDEX_NAME: round(sample_rate * 10e-3)}
    else:
        chunks = {}

    names = data.coords.keys() | data.keys() | data.indexes.keys()
    compressor = numcodecs.Blosc('zstd', clevel=6)

    for name in data.coords.keys():
        if data[name].dtype == np.dtype('object'):
            data[name] = data[name].astype('str')

    if mode == 'a':
        # follow existing encodings if appending
        encodings = None
    else:
        # despite that iq waveforms are the largest on disk,
        # compression tends to be slow and ineffective due
        # to high entropy
        encodings = {
            name: {'compressor': compressor}
            for name in names
            if name != waveform.iq_waveform.__name__
        }

    with zarr.storage.ZipStore(path, mode=mode, compression=0) as store:
        data.chunk(chunks).to_zarr(store, encoding=encodings)


def load(path: str | Path) -> xr.DataArray | xr.Dataset:
    """load a dataset or data array"""

    return xr.open_dataset(zarr.storage.ZipStore(path, mode='r'), engine='zarr')


def filter_iq_capture(
    iq: fourier.Array,
    capture: structs.FilteredCapture,
    *,
    axis=0,
    out=None,
):
    """apply a bandpass filter implemented through STFT overlap-and-add.

    Args:
        iq: the input waveform
        capture: the capture filter specification structure
        axis: the axis of `x` along which to compute the filter
        out: None, 'shared', or an array object to receive the output data

    Returns:
        the filtered IQ capture
    """

    xp = fourier.array_namespace(iq)

    fft_size = capture.analysis_filter['fft_size']

    fft_size_out, noverlap, overlap_scale, _ = fourier._ola_filter_parameters(
        iq.size,
        window=capture.analysis_filter['window'],
        fft_size_out=capture.analysis_filter.get('fft_size_out', fft_size),
        fft_size=fft_size,
        extend=True,
    )

    w = fourier._get_window(
        capture.analysis_filter['window'], fft_size, fftbins=False, xp=xp
    )
    freqs, _, xstft = fourier.stft(
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
        * fourier.equivalent_noise_bandwidth(w, fft_size, fftbins=False)
    )
    passband = (
        -capture.analysis_bandwidth / 2 + enbw,
        capture.analysis_bandwidth / 2 - enbw,
    )

    if fft_size_out != capture.analysis_filter['fft_size']:
        freqs, xstft = fourier.downsample_stft(
            freqs,
            xstft,
            fft_size_out=fft_size_out,
            passband=passband,
            axis=axis,
            out=xstft,
        )

    fourier.zero_stft_by_freq(freqs, xstft, passband=passband, axis=axis)

    return fourier.istft(
        xstft,
        iq.shape[axis],
        fft_size=fft_size_out,
        noverlap=noverlap,
        out=out,
        axis=axis,
    )


def simulated_awgn(
    capture: structs.Capture, *, power: float = 1, xp=np, pinned_cuda=False, out=None
):
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
