from __future__ import annotations
import numpy as np
from typing import Any
from iqwaveform import fourier
from functools import wraps
from typing import Optional


class WaveformSource:
    def __init__(
        self,
        sample_rate: float,
        analysis_bandwidth: Optional[float] = None,
        buffer: Any = None,
        analysis_filter: dict = {'fft_size': 1024, 'window': 'hamming'},
    ):
        self.sample_rate = sample_rate
        self.analysis_bandwidth = analysis_bandwidth
        self.buffer = buffer
        self.analysis_filter = analysis_filter

    def build_metadata(self) -> dict:
        return {
            'sample_rate': self.sample_rate,
            'analysis_bandwidth': self.analysis_bandwidth,
            'analysis_filter': dict(self.analysis_filter)
            if self.analysis_bandwidth
            else None,
        }

    def build_index_variables(self) -> dict:
        return {}

    def __hash__(self):
        """a hash that is unique from the perspective of sampling constants.

        This allows use of WaveformSource objects as arguments to functions
        defined with @lru_cache.
        """
        return hash((self.sample_rate, self.analysis_bandwidth))


def optional_filter(decorated):
    @wraps(decorated)
    def func(source, *args, **kws):
        iq = decorated(source, *args, **kws)

        if source.analysis_bandwidth is None:
            return iq

        return fourier.ola_filter(
            iq,
            fs=source.sample_rate,
            passband=(-source.analysis_bandwidth / 2, source.analysis_bandwidth / 2),
            fft_size=source.analysis_filter['fft_size'],
            window=source.analysis_filter['window'],
            out=source.buffer,
        )

    return func


@optional_filter
def simulated_awgn(
    source: WaveformSource,
    duration: float,
    *,
    power: float = 1,
    xp=np,
    pinned_cuda=False,
):
    try:
        # e.g., numpy
        bitgen = xp.random.PCG64()
    except AttributeError:
        # e.g., cupy
        bitgen = xp.random.MRG32k3a()

    generator = xp.random.Generator(bitgen)
    size = int(duration * source.sample_rate)

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

    if source.analysis_bandwidth is not None:
        power = power * source.sample_rate / source.analysis_bandwidth

    samples *= xp.sqrt(power / 2)

    return samples
