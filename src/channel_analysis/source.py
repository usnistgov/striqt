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

    names = data.coords.keys()|data.keys()|data.indexes.keys()
    compressor = numcodecs.Blosc('zstd', clevel=6)

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


def filter_after(decorated_func: callable):
    """apply a filter after the decorated function if a structs.FilteredCapture is passed"""

    @wraps(decorated_func)
    def func(capture, *args, out=None, **kws):
        iq = decorated_func(capture, *args, out=out, **kws)

        if (
            isinstance(capture, structs.FilteredCapture)
            or capture.analysis_bandwidth is None
        ):
            return iq

        return fourier.ola_filter(
            iq,
            fs=capture.sample_rate,
            passband=(-capture.analysis_bandwidth / 2, capture.analysis_bandwidth / 2),
            fft_size=capture.analysis_filter['fft_size'],
            window=capture.analysis_filter['window'],
            out=out,
        )

    return func


@filter_after
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
