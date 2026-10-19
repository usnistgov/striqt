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
    """evaluate the power spectrogram of x with the given arguments.

    The output is scaled such that the noise bandwidth is equal to the
    frequency resolution.
    """
    ret = stft(norm='power', **dict(locals()))
    if return_axis_arrays:
        freqs, times, X = ret
        return freqs, times, power.envtopow(X)
    else:
        return power.envtopow(ret)
