import numpy as np
import iqwaveform


def simulated_awgn(duration: float, sample_rate: float, power: float = 1, xp=np):

    try:
        # e.g., numpy
        bitgen = xp.random.PCG64()
    except AttributeError:
        # e.g., cupy
        bitgen = xp.random.MRG32k3a()

    generator = xp.random.Generator(bitgen)
    size = int(duration * sample_rate)

    samples = generator.standard_normal(size=2 * size, dtype=xp.float32).view(
        xp.complex64
    )

    samples *= xp.sqrt(power/2)

    return samples
