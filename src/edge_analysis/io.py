import numpy as np
import iqwaveform


def simulated_awgn(duration: float, sample_rate: float, power: float = 1):
    generator = np.random.Generator(np.random.PCG64())
    size = int(duration * sample_rate)

    samples = generator.standard_normal(size=2 * size, dtype=np.float32).view(
        np.complex64
    )

    samples *= np.sqrt(power / np.sqrt(2))

    return samples
