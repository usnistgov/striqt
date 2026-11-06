"""Power spectral estimates built on the STFT."""

from __future__ import annotations as __

import typing

from .. import power
from .stft import stft

if typing.TYPE_CHECKING:
    from ..typing import _AT, WindowType


@typing.overload
def spectrogram(
    x: _AT,
    *,
    fs: float,
    window: WindowType,
    nperseg: int,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    return_axis_arrays: typing.Literal[False],
) -> _AT: ...


@typing.overload
def spectrogram(
    x: _AT,
    *,
    fs: float,
    window: WindowType,
    nperseg: int,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    return_axis_arrays: typing.Literal[True] = True,
) -> tuple[_AT, _AT, _AT]: ...


@typing.overload
def spectrogram(
    x: _AT,
    *,
    fs: float,
    window: WindowType,
    nperseg: int,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    return_axis_arrays: bool,
) -> _AT | tuple[_AT, _AT, _AT]: ...


def spectrogram(
    x: _AT,
    *,
    fs: float,
    window: WindowType,
    nperseg: int,
    noverlap: int = 0,
    nzero: int = 0,
    axis: int = 0,
    truncate: bool = True,
    return_axis_arrays: bool = True,
) -> _AT | tuple[_AT, _AT, _AT]:
    """return the power spectrogram of `x` along `axis`.

    The arguments are those of `stft`. The noise bandwidth of the output equals the
    frequency resolution.

    Returns:
        ``(freqs, times, S)``, with `freqs` and `times` the axes of `stft` and `S`
        the linear power of each bin in the squared units of `x`; `S` alone when
        `return_axis_arrays` is False
    """
    ret = stft(norm='power', **dict(locals()))
    if return_axis_arrays:
        freqs, times, X = ret
        return freqs, times, power.envtopow(X)
    else:
        return power.envtopow(ret)
