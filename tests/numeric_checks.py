"""roundoff models, tolerance helpers and comparison assertions shared by the waveform
tests.

Every tolerance here is derived from the unit roundoff of the working dtype and a
count of the roundings the tested code performs, so that a failure means the library
rounds worse than its model rather than worse than a hand-picked number. The `*_tol`
helpers return ``{'rtol': ..., 'atol': ...}`` for splatting into `assert_close`.

Not a conftest: importable by bare name from every test module.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

# %% dtype helpers

FLOAT_DTYPES = (np.float32, np.float64)


def unit_roundoff(dtype):
    """u = eps / 2 of the real dtype underlying `dtype` (complex dtypes included)"""
    return np.finfo(dtype).eps / 2


def dtype_id(dtype):
    """parametrize id for a dtype argument, e.g. 'float32'"""
    return np.dtype(dtype).name


def func_id(value):
    """parametrize id for a function argument; other argument types get the default"""
    return getattr(value, '__name__', None)


def by_dtype(dtype, *, float32, float64):
    """the value matching the precision of `dtype` (complex dtypes select by their
    real component)"""
    return {'float32': float32, 'float64': float64}[np.finfo(dtype).dtype.name]


def rms(x, axis=None):
    """root-mean-square of `x` over `axis`; a float when axis is None"""
    out = np.sqrt(np.mean(np.abs(x) ** 2, axis=axis))
    return float(out) if axis is None else out


def reference_power(x, axis=None):
    """|x|**2 evaluated in complex128, averaged over `axis` when one is given"""
    power = np.abs(to_numpy(x).astype(np.complex128)) ** 2
    return power if axis is None else power.mean(axis=axis)


# %% array namespace conversion


def to_numpy(arr):
    """a numpy copy of a numpy, cupy or dask array, or of each array in a tuple or
    list of them"""
    if isinstance(arr, (tuple, list)):
        return type(arr)(to_numpy(a) for a in arr)
    if hasattr(arr, 'get'):  # cupy
        return arr.get()
    elif hasattr(arr, 'compute'):  # dask
        return arr.compute()
    return np.asarray(arr)


def numpy_and_cupy(cp, func, *args, xp_kwarg=None, **kws):
    """`func` evaluated on numpy arguments and again on their cupy copies.

    With `xp_kwarg`, the cupy call also receives the cupy module as that keyword, for
    functions that take the array namespace explicitly. Returns the two results as
    numpy arrays (or tuples of them) for comparison.
    """
    result_np = func(*args, **kws)
    cp_args = [cp.asarray(a) if isinstance(a, np.ndarray) else a for a in args]
    cp_kws = dict(kws, **{xp_kwarg: cp}) if xp_kwarg is not None else kws
    return result_np, to_numpy(func(*cp_args, **cp_kws))


# %% comparison assertions


def assert_close(
    actual, expected, *, rtol=0.0, atol=0.0, sigma=None, scale=None, err_msg=''
):
    """assert_allclose after to_numpy() on both sides.

    With `sigma`, an rms bound relative to `scale` (the rms of `expected` unless
    given), also require rms(actual - expected) < sigma * scale, and widen `atol` to
    at least the peak that `sigma` implies over the output size,
    peak_factor(size) * sigma * scale.
    """
    actual = to_numpy(actual)
    expected = to_numpy(expected)
    if sigma is not None:
        if scale is None:
            scale = rms(expected)
        rms_err = rms(actual - expected)
        assert rms_err < sigma * scale, (
            f'{err_msg} rms deviation {rms_err:.3e} above {sigma * scale:.3e} '
            f'({level_tolerance_dB(sigma):.1e} dB)'
        )
        atol = max(atol, peak_factor(expected.size) * sigma * scale)
    assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)


# %% FFT roundoff model (fourier, ofdm)
#
# Roundoff model for the cross-backend and far-bin tests. Per FFT pass the rms error
# relative to the output rms is c*eps*sqrt(log2 N), eps the unit roundoff (Gentleman &
# Sande 1966; FFTW accuracy notes), and each elementwise rounding adds (eps/sqrt(3))**2
# of error variance. c was fitted with chores/tests/measure_fft_accuracy.py: pocketfft
# gives 0.55-0.65 (1.2 for sizes with a prime factor >= 128); cuFFT on a Jetson TX2i
# reaches 1.9 at N=512 and 2.1 for Bluestein sizes. Errors of independent backends add
# in quadrature (measured 0.83-1.0).
FFT_ROUNDOFF_C = 2.2
FFT_ROUNDOFF_SAFETY = 3
# An FFT of a tone concentrates roundoff in a few bins instead of spreading it evenly:
# measured up to 6.7 units of roundoff of the tone amplitude at the tone (cuFFT, N=512)
# and 3.9 in a far bin (cuFFT, N=1024), against ~2 for pocketfft.
TONE_PEAK_ROUNDOFF = 8

# elementwise tolerances for the deterministic outputs (windows, frequency axes)
RTOL_FLOAT32 = 1e-5
RTOL_FLOAT64 = 1e-12
ATOL = 1e-10

# Overlap-add reconstruction of a tone is limited by the COLA window's passband
# ripple after spectral truncation, not by roundoff. Measured for a hamming window at
# nfft=256: the rms level is within 0.015 dB and the per-sample envelope ripple within
# 0.25 dB (0.47 dB for a tone in the FIR transition band). Engineering bounds with margin:
COLA_LEVEL_DB = 0.05
COLA_RIPPLE_DB = 0.5

# scipy.signal.firwin2 with a hamming window reaches about -53 dB in the stopband,
# and its passband ripple is far below that; bound both at -40 dB
FIR_LEAKAGE = 1e-2
# _stft_fir_lowpass designs its FIR with a rectangular window, whose Gibbs sidelobes
# limit the stopband to about -21 dB
FIR_RECT_STOPBAND_DB = -20


def elementwise_rtol(dtype):
    """rtol for deterministic elementwise outputs (windows, frequency axes)"""
    return by_dtype(dtype, float32=RTOL_FLOAT32, float64=RTOL_FLOAT64)


def roundoff_rms(dtype, nffts, n_elementwise=0):
    """expected rms roundoff error of one backend, relative to the output rms"""
    eps = unit_roundoff(dtype)
    var = n_elementwise * (eps / np.sqrt(3)) ** 2
    var += sum((FFT_ROUNDOFF_C * eps * np.sqrt(np.log2(n))) ** 2 for n in nffts)
    return float(np.sqrt(var))


def single_backend_rms(dtype, nffts, n_elementwise=0):
    """rms tolerance on one backend's error against an exact reference, relative to
    the output rms"""
    return FFT_ROUNDOFF_SAFETY * roundoff_rms(dtype, nffts, n_elementwise)


def cross_backend_rms(dtype, nffts, n_elementwise=0):
    """rms tolerance on the difference between two backends, relative to output rms"""
    return np.sqrt(2) * single_backend_rms(dtype, nffts, n_elementwise)


def peak_factor(size):
    """max/rms ratio of `size` complex gaussian errors, with 2x margin on the tail"""
    return 2 * np.sqrt(np.log(size))


def rms_tolerance_dBc(sigma):
    """express an rms amplitude tolerance relative to the output rms as error power
    relative to the signal, in dBc"""
    return 20 * np.log10(sigma)


def level_tolerance_dB(sigma, power=False):
    """express a relative tolerance on an output as the uncertainty of its level in dB.

    `sigma` bounds an amplitude ratio (level 20*log10|x|) unless `power` is True
    (level 10*log10 x, e.g. spectrogram bins).
    """
    return (10 if power else 20) * np.log10(1 + sigma)


def tone_peak_roundoff(dtype):
    """bound on structured roundoff in any one bin, relative to a tone's amplitude"""
    return FFT_ROUNDOFF_SAFETY * TONE_PEAK_ROUNDOFF * unit_roundoff(dtype)


def far_bin_floor_dBc(sigma, nfft, size=None, dtype=np.complex64):
    """express the roundoff tolerance in bins away from a bin-centered tone, relative
    to the tone, in dBc.

    The tone occupies one bin while roundoff spreads evenly over all `nfft` bins. With
    `size`, the result is the peak tolerance over that many far bins, which is the
    larger of the white-noise tail and the structured `tone_peak_roundoff`.
    """
    if size is None:
        return rms_tolerance_dBc(sigma / np.sqrt(nfft))
    white = peak_factor(size) * sigma / np.sqrt(nfft)
    return rms_tolerance_dBc(max(white, tone_peak_roundoff(dtype)))


def tone_bin(nfft, bin_fraction):
    """the signed bin index k in [-nfft//2, nfft//2) at `bin_fraction` of the grid"""
    return round(bin_fraction * (nfft - 1)) - nfft // 2


def bin_centered_tone(nfft, k, nseg, dtype=np.complex64):
    """a unit tone with `k` cycles per segment, over `nseg` segments"""
    n = np.arange(nseg * nfft)
    return np.exp(2j * np.pi * k * n / nfft).astype(dtype)


def unit_tone(size, fs, f0, dtype=np.complex64):
    """a unit complex tone at f0 Hz sampled at fs Hz"""
    t = np.arange(size) / fs
    return np.exp(2j * np.pi * f0 * t).astype(dtype)


def tone_frequency(x, fs):
    """the frequency of the strongest spectral line of x, to within fs/x.size"""
    X = np.fft.fft(np.asarray(x).astype(np.complex128))
    return float(np.fft.fftfreq(x.size, 1 / fs)[np.argmax(np.abs(X))])


def interior(x, pad, axis=-1):
    """x with `pad` samples trimmed from each end of `axis`"""
    return np.moveaxis(np.moveaxis(np.asarray(x), axis, 0)[pad:-pad], 0, axis)


def level_error_dB(x, ref=1.0):
    return 20 * np.log10(np.abs(x) / ref)


def assert_tone_level(y, ref=1.0):
    """check the rms level and per-sample envelope of a reconstructed unit tone"""
    assert abs(level_error_dB(rms(y), ref)) < COLA_LEVEL_DB
    assert np.abs(level_error_dB(y, ref)).max() < COLA_RIPPLE_DB


# %% ulp budgets for the dB conversions (power_analysis, jit)
#
# Roundoff budgets for the dB conversions, in ulps per library call (1 ulp <= 2u
# relative, u = unit roundoff). Sized to cover CUDA's single-precision bounds (log10f 2,
# powf 8, hypotf 3 ulp) as well as libm (~1 ulp). Measured with
# chores/tests/measure_db_accuracy.py: libm log10 1.1-1.9 ulp; CUDA libdevice on a
# Jetson TX2i log10f 2.0, powf 5 (|x| <= 30 dB), complex |z| 1.6 input ulps.
LOG10_ULP = 2
POW_ULP = 8
HYPOT_ULP = 3
ROUNDOFF_SAFETY = 2
DB_PER_NEPER = 10 / np.log(10)


def log_conversion_tol(dtype, scale, complex_input=False, n_impl=1):
    """tolerances for scale*log10(|x|), against exact math (n_impl=1) or a second
    implementation (n_impl=2).

    Roundoff on the input side of the log becomes an absolute dB error, which is what
    bounds the result near 0 dB where ulps of the output are meaningless.
    """
    u = unit_roundoff(dtype)
    rtol = 2 * u * (LOG10_ULP + 0.5)
    input_ulp = (HYPOT_ULP if complex_input else 0) + 0.5
    atol = (scale / np.log(10)) * 2 * u * input_ulp
    return {
        'rtol': ROUNDOFF_SAFETY * n_impl * rtol,
        'atol': ROUNDOFF_SAFETY * n_impl * atol,
    }


def pow_conversion_rtol(dtype, max_abs_dB, n_impl=1):
    """rtol for 10**(x/10) with |x| <= max_abs_dB.

    Rounding x/10 perturbs the exponent, so its effect scales with |x|.
    """
    u = unit_roundoff(dtype)
    rtol = u * (np.log(10) * max_abs_dB / 10 + 2 * POW_ULP)
    return ROUNDOFF_SAFETY * n_impl * rtol


def envelope_power_rtol(dtype, complex_input=False, n_impl=1):
    """rtol for |x|**2"""
    u = unit_roundoff(dtype)
    rtol = 2 * u * (0.5 + (2 * HYPOT_ULP if complex_input else 0))
    return ROUNDOFF_SAFETY * n_impl * rtol


def linear_stat_tol(dtype, max_abs_dB, n, n_impl=1):
    """tolerances in dB for dBlinmean/dBlinsum over n terms with |x| <= max_abs_dB"""
    u = unit_roundoff(dtype)
    rel_linear = pow_conversion_rtol(dtype, max_abs_dB) + ROUNDOFF_SAFETY * n * u
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': n_impl * tol['rtol'],
        'atol': n_impl * (tol['atol'] + DB_PER_NEPER * rel_linear),
    }


def roundtrip_power_rtol(dtype, max_abs_dB):
    """rtol for dBtopow(powtodB(x)) with |powtodB(x)| <= max_abs_dB"""
    tol = log_conversion_tol(dtype, 10)
    dB_err = tol['rtol'] * max_abs_dB + tol['atol']
    return np.log(10) / 10 * dB_err + pow_conversion_rtol(dtype, max_abs_dB)


def roundtrip_dB_tol(dtype, max_abs_dB):
    """tolerances for powtodB(dBtopow(x)) with |x| <= max_abs_dB"""
    tol = log_conversion_tol(dtype, 10)
    return {
        'rtol': tol['rtol'],
        'atol': tol['atol'] + DB_PER_NEPER * pow_conversion_rtol(dtype, max_abs_dB),
    }


def linear_tolerance_dB(rtol):
    """express a relative tolerance on a linear power as a tolerance in dB"""
    return 10 * np.log10(1 + rtol)


def dB_tolerance(rtol, atol, max_abs_dB):
    """express (rtol, atol) on a dB-valued output as its worst-case tolerance in dB"""
    return atol + rtol * max_abs_dB


# %% binned statistics (arrays, power_analysis)


def accum_rtol(dtype, n, n_impl=1):
    """rtol on a sum or reduction over `n` terms of `dtype`, against exact arithmetic
    (n_impl=1) or against a second implementation (n_impl=2)"""
    return ROUNDOFF_SAFETY * n_impl * n * unit_roundoff(dtype)


def mean_atol(x, count):
    """absolute roundoff bound on the mean of `count` samples drawn from `x`"""
    return count * np.finfo(x.dtype).eps * float(np.abs(x).max())


def reference_binned_mean(x, count, axis):
    """mean over contiguous, left-aligned bins of `count` along `axis`"""
    moved = np.moveaxis(x, axis, -1)
    m = moved.shape[-1] // count
    binned = moved[..., : m * count].reshape(moved.shape[:-1] + (m, count))
    return np.moveaxis(binned.mean(axis=-1), -1, axis)


def blocks(x, size, axis):
    """the axis of x reshaped into (n_blocks, size) after truncating the remainder"""
    x = np.moveaxis(x, axis, -1)
    n_blocks = x.shape[-1] // size
    x = x[..., : n_blocks * size].reshape(x.shape[:-1] + (n_blocks, size))
    return np.moveaxis(x, (-2, -1), (axis, axis + 1))


# %% cyclic-prefix correlation (jit, ofdm)


def corr_atol(x, n_inds, norm, n_impl=1):
    """absolute tolerance on _corr_at_indices against exact arithmetic.

    Each of the n_inds products a*conj(b) and the power terms are rounded at the input
    precision before the complex128 accumulation, and the result is rounded once more
    on output. With norm=True the Cauchy-Schwarz bound sum|a||b| <= sqrt(Pa*Pb) makes
    the error relative to a unit-scale output; with norm=False it is relative to the
    largest product magnitude.
    """
    from striqt.waveform.lib.arrays import float_dtype_like

    u = unit_roundoff(float_dtype_like(x))
    if norm:
        scale = 1.0
    else:
        scale = float(np.abs(x).max() ** 2)
    return ROUNDOFF_SAFETY * n_impl * (n_inds + 3) * u * scale
