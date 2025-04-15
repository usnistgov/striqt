import functools
import typing
from ..api import util

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')


@functools.lru_cache()
def _m_sequence(N_id2: int) -> list[int]:
    """compute the M-sequence used as the 5G-NR primary synchronization sequence.

    These express frequency-domain values of the active subcarriers, spaced at the
    subcarrier spacing.

    Args:
        N_id2: one of (0,1,2), expressing the sector portion of the cell ID
    """
    x = [0, 1, 1, 0, 1, 1, 1]

    for i in range(7, 127):
        x.append((x[i - 3] + x[i - 7]) % 2)

    m = [(n + 43 * N_id2) % 127 for n in range(127)]

    pss = [(1 - 2 * x[_m]) for _m in m]

    return pss


@functools.lru_cache()
def pss_5g_nr(
    sample_rate: float,
    subcarrier_spacing: float,
    center_frequency=0,
    *,
    xp=np,
    dtype='complex64',
):
    """compute the PSS correlation sequences at the given sample rate for each N_id2.

    The sequence can be convolved with an IQ waveform of the same sample rate
    along the last axis to compute a synchronization correlation sequence. The
    result would be normalized to the IQ input power.

    Args:
        sample_rate: the desired output sample rate (in S/s), a multiple of subcarrier_spacing and at least (127*subcarrier_spacing)
        subcarrier_spacing: the subcarrier spacing (in Hz), a multiple of 15e3

    Returns:
        xp.ndarray with dimensions (N_id2 index, PSS sample index)
    """

    # number of occupied subcarriers in the PSS
    SC_COUNT = 127

    if not iqwaveform.util.isroundmod(subcarrier_spacing, 15e3):
        raise ValueError('subcarrier_spacing must be a multiple of 15000')

    min_sample_rate = SC_COUNT * subcarrier_spacing
    if sample_rate < min_sample_rate:
        raise ValueError(f'sample_rate must be at least {min_sample_rate} S/s')

    if iqwaveform.util.isroundmod(sample_rate, subcarrier_spacing):
        size_out = round(sample_rate / subcarrier_spacing)
    else:
        raise ValueError('sample_rate must be a multiple of subcarrier spacing')

    if center_frequency == 0:
        frequency_offset = 0
    elif iqwaveform.util.isroundmod(center_frequency, subcarrier_spacing):
        # check frequency bounds later via pad_*
        frequency_offset = round(center_frequency / subcarrier_spacing)
    else:
        raise ValueError(
            'center_frequency must be a whole multiple of subcarrier_spacing'
        )

    if size_out == SC_COUNT and frequency_offset == 0:
        pad_left = 0
        pad_right = 0
    else:
        pad_left = size_out // 2 - 120 + 56 + frequency_offset
        pad_right = size_out - SC_COUNT - pad_left

    if pad_left < 0 or pad_right < 0:
        raise ValueError(
            'center_frequency shift pushes M-sequence outside of Nyquist sample rate'
        )

    m_sequences = np.array([_m_sequence(i) for i in range(3)], dtype=dtype)
    norm = np.float32(np.sqrt(SC_COUNT))
    pss_freq = iqwaveform.util.pad_along_axis(
        m_sequences / norm, [(pad_left, pad_right)], axis=1
    )
    pss_time = np.fft.ifft(np.fft.fftshift(pss_freq, axes=1), axis=1)

    # prepend the cyclic prefix
    cp_size = round(9 * sample_rate / subcarrier_spacing / 128)
    pss_time = np.concatenate([pss_time[:, -cp_size:], pss_time], axis=1)

    return xp.array(pss_time)


def pss_offset(symbol_offset, sample_rate, subcarrier_spacing):
    sc_count = sample_rate / subcarrier_spacing
    spacing = round(
        (1 + symbol_offset) * sc_count + (10 + (symbol_offset + 1) * 9) / 128 * sc_count
    )
    return spacing + round(sc_count / 128)
