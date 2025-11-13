"""Some window functions not included by scipy.signal"""

from .util import lazy_import
import typing


if typing.TYPE_CHECKING:
    import numpy as np
    from scipy import signal
else:
    signal = lazy_import('signal', package='scipy')
    np = lazy_import('numpy')


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


def knab(M: int, alpha, sym=True) -> 'np.ndarray':
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


def cosh(M: int, alpha, sym=True) -> 'np.ndarray':
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
    registry = signal.windows._windows._win_equiv
    registry['acg'] = acg
    registry['cosh'] = cosh
    registry['modified_bessel'] = modified_bessel
    registry['knab'] = knab
