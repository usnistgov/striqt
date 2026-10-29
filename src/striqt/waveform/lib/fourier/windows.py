"""Window design for spectral analysis: the cached `get_window` family and window
functions not included in `scipy.signal`."""

from __future__ import annotations as __

import typing

from .. import util
from ..arrays import convert_np_to_xp, dtype_change_float
from ..util import np, scipy

if typing.TYPE_CHECKING:
    from ..typing import Array, DTypeLike, WindowSpecType, WindowType, XpType


# %% Extra windows
def _len_guards(M):
    """return True when the window length `M` is 0 or 1.

    Raises:
        ValueError: if `M` is not a non-negative integer
    """
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
    """return the approximate confined Gaussian (ACG) window of `M` samples.

    The ACG window approximates the DPSS (Slepian) window.

    Args:
        M: the window length (in samples)
        sigma_t: the rms temporal width of the window as a fraction of `M`
        sym: True for a symmetric window; False for a periodic (DFT-even) window
        dtype: the floating-point dtype of the result

    Returns:
        a 1-D numpy array of `M` samples with peak value 1

    Raises:
        ValueError: if `M` is not a non-negative integer

    References:
        S. Starosielec and D. Hägele, "Discrete-time windows with minimal RMS
        bandwidth for given RMS temporal width," Signal Processing, vol. 102,
        pp. 240-246, Sept. 2014.
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
    """register the ``'acg'``, ``'cosh'``, ``'modified_bessel'``, and ``'knab'`` windows with scipy.

    The registration modifies the window table of `scipy.signal.get_window` for the
    process.
    """
    from scipy.signal.windows import _windows

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
            # the shift is [1, -1, 1, -1, ...] for even sizes
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
    xp: XpType = None,
) -> Array:
    """return a window of `nwindow` samples for `xp`, zero-padded to `nwindow + nzero`.

    The design is cached in memory and on disk. A call that misses both runs the
    scipy window design and writes to the disk cache. Equivalent spellings of the
    arguments (a dtype given as a string or a `numpy.dtype`, `xp` given or omitted,
    positional or keyword `nzero`) share one cache entry. A numpy result is the
    cached array and must not be modified in place.

    Args:
        name_or_tuple: a window name, ``(name, parameter)`` tuple, or Kaiser beta
            accepted by `scipy.signal.get_window`, or a name registered by
            `register_extra_windows`
        nwindow: the number of nonzero window samples
        nzero: the number of zero samples appended to the window
        fftshift: True to modulate the window by a half-band shift so that the FFT of
            a windowed signal is in ascending frequency order; the window is
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
    """return the equivalent noise bandwidth (ENBW) of a window, in bins.

    One bin is ``fs / N`` wide for a window of `N` nonzero samples (Harris 1978). The
    ENBW in Hz is the returned value times ``fs / N``. An FFT of the window
    zero-padded to ``N + nzero`` samples has bins of ``fs / (N + nzero)``. The same
    ENBW spans ``(N + nzero) / N`` times as many of those bins.

    Args:
        window: the window name or ``(name, parameter)`` for `get_window`
        N: the number of nonzero window samples
        fftbins: the `get_window` periodic-window flag
        cached: True to design through the cached `get_window`

    Returns:
        the ENBW as a float: 1.0 for a rectangular window and larger for every other

    References:
        F. J. Harris, "On the use of windows for harmonic analysis with the discrete
        Fourier transform," Proc. IEEE, vol. 66, no. 1, pp. 51-83, Jan. 1978.
    """
    if cached:
        w = get_window(window, N, fftbins=fftbins)
    else:
        w = _design_window(window, N, fftbins=fftbins)
    return len(w) * np.sum(w**2) / np.sum(w) ** 2


@util.lru_cache()
def find_window_param_from_enbw(
    window_name: str, target: float, *, nfft: int = 4096, atol=1e-6
) -> float:
    """return the parameter of `window_name` with equivalent noise bandwidth `target`.

    The parameter is found for a window of `nfft` samples and converges to a limit as
    `nfft` grows. For `target` of at least 1.1, the result is slightly below
    ``target**2`` for ``'dpss'`` and slightly below ``pi * target**2`` for
    ``'kaiser'``.

    Args:
        window_name: ``'kaiser'``, ``'dpss'``, or ``'chebwin'``
        target: the equivalent noise bandwidth (in bins), above ``1 + 1/nfft``
        nfft: the window length (in samples) at which the ENBW is evaluated
        atol: the absolute tolerance on the parameter

    Returns:
        the parameter for ``get_window((window_name, result), ...)``

    Raises:
        ValueError: if `target` is at most ``1 + 1/nfft``, or if `window_name` is not
            one of the three above
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
