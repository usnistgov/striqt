from __future__ import annotations
import numpy as np
from iqwaveform import fourier
from functools import wraps
from . import structs


def filter_after(decorated):
    """apply a filter after the decorated function if a structs.FilteredCapture is passed"""

    @wraps(decorated)
    def func(capture, *args, out=None, **kws):
        iq = decorated(capture, *args, out=out, **kws)

        if isinstance(capture, structs.FilteredCapture) or capture.analysis_bandwidth is None:
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
    capture: structs.Capture,
    *,
    power: float = 1,
    xp=np,
    pinned_cuda=False,
    out=None
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
