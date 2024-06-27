import numpy as np
import numba
import numba.cuda


def simulated_awgn(
    duration: float, sample_rate: float, power: float = 1, xp=np, pinned_cuda=False
):
    try:
        # e.g., numpy
        bitgen = xp.random.PCG64()
    except AttributeError:
        # e.g., cupy
        bitgen = xp.random.MRG32k3a()

    generator = xp.random.Generator(bitgen)
    size = int(duration * sample_rate)

    if pinned_cuda:
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

    samples *= xp.sqrt(power / 2)

    return samples
