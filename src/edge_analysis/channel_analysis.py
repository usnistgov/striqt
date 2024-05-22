from __future__ import annotations
import iqwaveform
from functools import lru_cache
from iqwaveform.power_analysis import iq_to_cyclic_power
import numpy as np
from scipy.signal import ellip, ellipord, sosfreqz
from iqwaveform.util import array_namespace

@lru_cache(8)
def generate_iir_lpf(
    rp_dB: (float|int),
    rs_dB: (float|int),
    cutoff_Hz: (float|int),
    width_Hz: (float|int),
    fs: (float|int),
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter.

    Parameters
    ----------
    rp_dB: (float, int)
        Maximum passband ripple below unity gain, in dB.
    rs_dB: (float, int)
        Minimum stopband attenuation, in dB.
    cutoff_Hz: (float, int)
        Filter cutoff frequency, in Hz.
    width_Hz: (float, int)
        Passband-to-stopband transition width, in Hz.
    Fs_MHz: (float, int)
        Sampling rate, in MHz.
    plot_response: bool
        If True, plot the filter response.

    Returns
    -------
    sos: numpy.ndarray
        Second-order sections representation of the IIR filter.
    """

    # Generate filter
    ord, wn = ellipord(cutoff_Hz, cutoff_Hz + width_Hz, rp_dB, rs_dB, False, fs)
    sos = ellip(ord, rp_dB, rs_dB, wn, "lowpass", False, "sos", fs)

    return sos

@lru_cache(16)
def _get_apd_bins(lo, hi, count):
    return np.linspace(lo, hi, count)

def power_time_series(iq, *, fs: float, detector_period: float, detectors=('rms', 'peak')):
    return {
        detector: iqwaveform.iq_to_bin_power(
            iq, Ts=1/fs, Tbin=detector_period, kind=detector
        )
        for detector in detectors
    }

def amplitude_probability_distribution(iq, *, power_low, power_high, power_count):
    bins = _get_apd_bins(power_low, power_high, power_count)
    ccdf = iqwaveform.sample_ccdf(iqwaveform.envtodB(iq), bins)

    return ccdf, bins

def persistence_spectrum(iq, *, fs, window, fres, quantiles):
    # TODO: validate fs % fres
    # TODO: implement other statistics, such as mean
    xp = array_namespace(iq)
    fft_size = int(fs/fres)
    freqs, times, X = iqwaveform.stft(iq, window='flattop', fs=fs, nperseg=fft_size)
    spectrum = xp.quantile(iqwaveform.envtopow(X), quantiles, axis=0)
    return freqs, iqwaveform.powtodB(spectrum.T)