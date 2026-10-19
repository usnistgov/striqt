"""Window design for spectral analysis: the cached `get_window` family and window
functions not included in `scipy.signal`.

`register_extra_windows` enters the extra windows into scipy's lookup table, where
`scipy.signal.get_window` and `get_window` resolve them by name.
"""

from __future__ import annotations as __

import typing

from .. import util
from ..arrays import convert_np_to_xp, dtype_change_float
from ..util import np, scipy

if typing.TYPE_CHECKING:
    from ..typing import Array, DTypeLike, WindowSpecType, WindowType, XpType


# %% Extra windows
def _len_guards(M):
    """Handle small or incorrect window lengths"""
    if int(M) != M or M < 0:
        raise ValueError('Window length M must be a non-negative integer')
    return M <= 1


def _extend(M, sym):
    """Extend window by 1 sample if needed for DFT-even symmetry"""
    if not sym:
        return M + 1, True
    else:
        return M, False


def _truncate(w, needed):
    """Truncate window by 1 sample if needed for DFT-even symmetry"""
    if needed:
        return w[:-1]
    else:
        return w


def knab(M: int, alpha, sym=True) -> np.ndarray:
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = np.sinh((np.pi * alpha) * sqrt_term) / (np.sinh(np.pi * alpha) * sqrt_term)

    w[0] = w[-1] = np.pi * alpha / np.sinh(np.pi * alpha)
    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


def modified_bessel(M, alpha, sym=True):
    from scipy import special

    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = special.i1((np.pi * alpha) * sqrt_term) / (
        special.i1(np.pi * alpha) * sqrt_term
    )

    w[0] = w[-1] = 0  # np.pi*alpha/np.sinh(np.pi*alpha)

    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


def cosh(M: int, alpha, sym=True) -> np.ndarray:
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    t = np.linspace(-0.5, 0.5, M)

    sqrt_term = np.sqrt(1 - (2 * t) ** 2)
    w = np.cosh((np.pi * alpha) * sqrt_term) / (np.cosh(np.pi * alpha) * sqrt_term)

    w[0] = w[-1] = 1 / np.cosh(np.pi * alpha)

    w /= np.sqrt(np.sum(w**2))

    return _truncate(w, needs_trunc)


def acg(M: int, sigma_t: float, sym=True, dtype='float64'):
    """returns approximate confined gaussian window.

    In practice, this is a close approximation of the Slepian
    window.

    Args:
        M: window size, in samples
        sigma_t: The (3-dB) uncertainy resolution in time bins
    Reference:
        S. Starosielec, D. Hägele, "Discrete-time windows with minimal RMS bandwidth for given RMS temporal width,
        Signal Processing," Signal Processing Vol. 102, Sept. 2014, Pages 240-246.
    """

    if _len_guards(M):
        return np.ones(M)

    M, needs_trunc = _extend(M, sym)

    def G(k, sigma_t=sigma_t):
        inner = (k - (M - 1) / 2) / (2 * M * sigma_t)
        return np.exp(-(inner**2))

    k = np.arange(M, dtype=dtype)
    w = G(k) - G(-0.5) * (G(k + M) + G(k - M)) / (G(-0.5 + M) + G(-0.5 - M))
    w /= w.max()

    return _truncate(w, needs_trunc)


def register_extra_windows():
    """add 'acg', 'cosh', 'modified_bessel', 'knab', and 'taylor' windows to
    the window functions registered for access by `scipy.signal.get_window`.
    """
    import scipy.signal.windows._windows as _windows

    if hasattr(_windows, '_win_equiv'):
        registry: dict[str, typing.Callable] = _windows._win_equiv
    else:
        registry: dict[str, typing.Callable] = _windows._WIN_FUNCS  # type: ignore
    registry['acg'] = acg
    registry['cosh'] = cosh
    registry['modified_bessel'] = modified_bessel
    registry['knab'] = knab


# %% Cached window design
def _design_window(
    name_or_tuple: WindowType,
    nwindow: int,
    nzero: int = 0,
    *,
    fftshift: bool = False,
    center_zeros: bool = False,
    fftbins: bool = True,
    norm: bool = True,
    dtype: DTypeLike = 'float32',
) -> Array:
    """design the window of `get_window` in numpy, without caching"""

    from scipy import signal

    register_extra_windows()

    ws = signal.windows.get_window(name_or_tuple, nwindow, fftbins=fftbins)
    ws = typing.cast('Array', ws)

    ntotal = nwindow + nzero

    if nzero == 0:
        w = ws
    elif center_zeros:
        w = np.empty(ntotal, dtype=ws.dtype)
        w[nzero // 2 : nzero // 2 + nwindow] = ws
        w[: nzero // 2] = 0
        w[nzero // 2 + nwindow :] = 0
    else:
        w = np.empty(ntotal, dtype=ws.dtype)
        w[:nwindow] = ws
        w[nwindow:] = 0

    if norm:
        # scale the time-averaged power to 1
        w /= np.sqrt(np.mean(np.abs(w) ** 2))

    if fftshift:
        delay = scipy.ndimage.fourier_shift(np.ones_like(w), ntotal // 2)

        if ntotal % 2 == 0:
            # really just [1, -1, 1, -1, 1, ...]
            delay = delay.real

        w = delay * w

    if dtype is not None:
        dtype_out = dtype_change_float(w.dtype, dtype)
        w = w.astype(dtype_out)

    return w


_cached_design_window = util.lru_cache()(util.persistent_cache()(_design_window))


@convert_np_to_xp
def get_window(
    name_or_tuple: WindowType,
    nwindow: int,
    nzero: int = 0,
    *,
    fftshift: bool = False,
    center_zeros: bool = False,
    fftbins: bool = True,
    norm: bool = True,
    dtype: DTypeLike = 'float32',
    xp: 'XpType' = None,
) -> Array:
    """return a window of `nwindow` samples for `xp`, zero-padded to `nwindow + nzero`.

    The design is cached in memory and on disk (`persistent_cache`), so a call that
    misses both costs the scipy window design and a disk write. Equivalent spellings
    of the arguments (a dtype given as a string or a `numpy.dtype`, `xp` given or
    omitted, positional or keyword `nzero`) share one cache entry.

    Args:
        name_or_tuple: a window name, ``(name, parameter)`` tuple, or Kaiser beta
            accepted by `scipy.signal.get_window`, or a name registered by
            `register_extra_windows`
        nwindow: the number of nonzero window samples
        nzero: the number of zero samples appended to the window
        fftshift: True to modulate the window by a half-band shift so that the FFT of
            a windowed signal comes out in ascending frequency order; the window is
            then complex when `nwindow + nzero` is odd
        center_zeros: True to split the `nzero` zeros before and after the window
            instead of appending them
        fftbins: True for a periodic window (spectral analysis), False for a
            symmetric one, as in `scipy.signal.get_window`
        norm: True to scale the window to unit time-averaged power
        dtype: the floating-point precision of the result (the complex counterpart
            when `fftshift` makes the window complex); None keeps scipy's float64
        xp: the array namespace of the result; None selects numpy

    Returns:
        a 1-D array of `nwindow + nzero` samples in `xp`

    References:
        `scipy.signal.get_window`
    """
    if dtype is not None:
        dtype = np.dtype(dtype)

    return _cached_design_window(
        name_or_tuple,
        nwindow,
        nzero,
        fftshift=bool(fftshift),
        center_zeros=bool(center_zeros),
        fftbins=bool(fftbins),
        norm=bool(norm),
        dtype=dtype,
    )


typing.cast(typing.Any, get_window).cache_info = _cached_design_window.cache_info
typing.cast(typing.Any, get_window).cache_clear = _cached_design_window.cache_clear


@util.lru_cache()
def equivalent_noise_bandwidth(window: WindowSpecType, N, fftbins=True, cached=True):
    """return the equivalent noise bandwidth (ENBW) of a window, in bins"""
    if cached:
        w = get_window(window, N, fftbins=fftbins)
    else:
        w = _design_window(window, N, fftbins=fftbins)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@util.lru_cache()
def find_window_param_from_enbw(
    window_name: str, target: float, *, nfft: int = 4096, atol=1e-6
) -> float:
    """find the parameter that satistifes the specified equivalent-noise bandwidth (ENBW)
    for a given single-parameter window.

    The estimate is performed for the given `nfft`; when it is
    larger, the result will tend to converge toward a central value.

    Typically, where enbw is at least 1.1:
        * when `window_name == 'dpss'`: the result will be slightly smaller than `(enbw)**2`
        * when `window_name == 'kaiser'`: the result will be slightly smaller than `pi * (enbw)**2`

    Arguments:
        window_name: one of 'kaiser', 'dpss', or 'chebwin'
        enbw: the desired equivalent noise bandwidth (in FFT bins)
        nfft: the window size used to estimate ENBW
        atol: the absolute error tolerance in the estimated NW

    Returns:
        result parameter suited for `get_window((window_name, result), ...)`
    """
    from scipy.optimize import bisect

    if target < 1 + 1 / nfft:
        raise ValueError('enbw must be greater than 1')

    def err(x):
        enbw = equivalent_noise_bandwidth.__wrapped__(
            (window_name, x), nfft, cached=False
        )
        return enbw - target

    if window_name == 'kaiser':
        a = np.pi * 1e-2
        b = min(target**2, nfft // 2 - 1) * np.pi
    elif window_name == 'dpss':
        a = 1e-2
        b = min(target**2, nfft // 2 - 1)
    elif window_name == 'chebwin':
        a = 45
        b = 1000
    else:
        raise ValueError('window_name must be one of ("kaiser", "dpss", "chebwin")')

    result = bisect(err, a, b, xtol=atol)
    assert isinstance(result, float)

    return result
