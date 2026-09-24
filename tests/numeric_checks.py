"""comparison assertions, reference helpers and test-only tolerance policy shared by
the suite.

The roundoff models themselves live in the library beside the functions they describe
(`striqt.waveform.fourier`, `striqt.waveform.power_analysis`,
`striqt.waveform.lib.arrays`, `striqt.waveform.ofdm`) and the per-measurement budgets
are registered with `striqt.analysis.registry`; tests take their pass criteria from
there, so the model the suite enforces is the one the library reports.

Not a conftest: importable by bare name from every test module.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from striqt.waveform.lib.fourier import peak_factor
from striqt.waveform.power_analysis import level_tolerance_dB

# %% dtype helpers

FLOAT_DTYPES = (np.float32, np.float64)


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

    `atol` may be an array, for an output whose tolerance varies element by element
    (`striqt.analysis.util.elementwise_atol`); assert_allclose cannot take one, so the
    comparison is made here.
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
        atol = np.maximum(atol, peak_factor(expected.size) * sigma * scale)
    if np.ndim(atol) == 0:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)
    else:
        assert_within(actual, expected, atol + rtol * np.abs(expected), err_msg=err_msg)


def assert_within(actual, expected, tolerance, *, err_msg=''):
    """assert |actual - expected| <= `tolerance`, which may vary per element"""
    excess = np.abs(to_numpy(actual) - to_numpy(expected)) - tolerance
    i = tuple(int(j) for j in np.unravel_index(int(np.argmax(excess)), excess.shape))
    assert excess.max() <= 0, (
        f'{err_msg} {int((excess > 0).sum())}/{excess.size} elements outside '
        f'tolerance; worst at {i}: |{actual[i]:.8g} - {expected[i]:.8g}| exceeds '
        f'{np.broadcast_to(tolerance, excess.shape)[i]:.3e} by {excess.max():.3e}'
    )


def level_error_dB(x, ref=1.0):
    return 20 * np.log10(np.abs(x) / ref)


def assert_tone_level(y, ref=1.0):
    """check the rms level and per-sample envelope of a reconstructed unit tone"""
    assert abs(level_error_dB(rms(y), ref)) < COLA_LEVEL_DB
    assert np.abs(level_error_dB(y, ref)).max() < COLA_RIPPLE_DB


# %% measurement result views


def levels(result) -> np.ndarray:
    """the dB values of a measurement result, widened to float64 for comparison"""
    values = result.values if hasattr(result, 'values') else result
    return np.asarray(values, dtype='float64')


def populated(fractions, bins):
    """the (bin, fraction) pairs of the non-zero entries of one histogram row"""
    index = np.nonzero(np.asarray(fractions))
    return list(zip(bins[index], np.asarray(fractions)[index]))


# %% test-only tolerance policy

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


def cross_backend(numpy_budget, cupy_budget):
    """the tolerance on the difference between a numpy and a cupy evaluation, each
    within its own exact-math budget: independent roundoff adds in quadrature
    (measured 0.83-1.0 of it, chores/tests/measure_fft_accuracy.py).

    Takes two floats, or two `striqt.analysis.specs.Tolerance` of the same units, whose
    `off_peak_dBc` combine as the amplitude errors they encode.
    """
    if isinstance(numpy_budget, (int, float)):
        return float(np.hypot(numpy_budget, cupy_budget))
    assert numpy_budget.units == cupy_budget.units

    def in_quadrature(a, b):
        return a.replace(
            rms=float(np.hypot(a.rms, b.rms)), peak=float(np.hypot(a.peak, b.peak))
        )

    def depth_dBc(a, b):
        return float(20 * np.log10(np.hypot(10 ** (a / 20), 10 ** (b / 20))))

    depths = (numpy_budget.off_peak_dBc, cupy_budget.off_peak_dBc)
    if None in depths:
        off_peak = next((d for d in depths if d is not None), None)
    else:
        off_peak = depths[0].replace(
            rms=depth_dBc(depths[0].rms, depths[1].rms),
            peak=depth_dBc(depths[0].peak, depths[1].peak),
        )
    return numpy_budget.replace(
        rtol=float(np.hypot(numpy_budget.rtol, cupy_budget.rtol)),
        on_peak=in_quadrature(numpy_budget.on_peak, cupy_budget.on_peak),
        off_peak_dBc=off_peak,
    )


# %% reference signals and layouts


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
