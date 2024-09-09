from __future__ import annotations
from typing import Literal, get_args
from functools import lru_cache
import numpy as np
import numba as nb
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform
from .. import structs, dataarrays

ofdm = iqwaveform.util.lazy_import('iqwaveform.ofdm')
pd = iqwaveform.util.lazy_import('pandas')


### Time elapsed dimension and coordinates
CyclicSampleLagAxis = Literal['cyclic_sample_lag']


@dataclass
class CyclicSampleLagCoords:
    data: Data[CyclicSampleLagAxis, np.float32]
    standard_name: Attr[str] = 'Cyclic symbol index offset'
    # units: Attr[str] = 's'

    @staticmethod
    @lru_cache
    def factory(
        capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...], **_
    ) -> dict[str, np.ndarray]:
        max_len = _get_correlation_length(
            capture, subcarrier_spacings=subcarrier_spacings
        )
        return pd.RangeIndex(0, max_len, name=get_args(CyclicSampleLagAxis)[0])


### Subcarrier spacing label axis
SubcarrierSpacingAxis = Literal['subcarrier_spacing']


@dataclass
class SubcarrierSpacingCoords:
    data: Data[SubcarrierSpacingAxis, np.float32]
    standard_name: Attr[str] = 'Subcarrier spacing'
    units: Attr[str] = 'Hz'

    @staticmethod
    @lru_cache
    def factory(
        capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...], **_
    ):
        return tuple(subcarrier_spacings)


### Dataarray definition
@dataclass
class CellularCyclicAutocorrelation(AsDataArray):
    power_time_series: Data[
        tuple[SubcarrierSpacingAxis, CyclicSampleLagAxis], np.float32
    ]

    subcarrier_spacing: Coordof[SubcarrierSpacingCoords]
    cyclic_sample_lag: Coordof[CyclicSampleLagCoords]


### iqwaveform wrapper
@dataarrays.as_registered_channel_analysis(CellularCyclicAutocorrelation)
def cellular_cyclic_autocorrelation(
    iq,
    capture: structs.Capture,
    *,
    subcarrier_spacings: tuple[float, ...] = (15e3, 30e3, 60e3),
    frame_limit: int = 2,
    normalize: bool = True,
):
    xp = iqwaveform.util.array_namespace(iq)
    subcarrier_spacings = tuple(subcarrier_spacings)
    phy_scs = _get_phy_mappings(capture.analysis_bandwidth, subcarrier_spacings, xp=xp)

    kws = dict(
        slots='all', symbols='all', frames=np.arange(0, frame_limit), norm=normalize
    )

    max_len = _get_correlation_length(capture, subcarrier_spacings=subcarrier_spacings)

    result = xp.full((len(subcarrier_spacings), max_len), np.nan, dtype=np.float32)
    for i, phy in enumerate(phy_scs.values()):
        R, _ = _correlate_cyclic_prefixes(iq, phy, **kws)
        result[i][: R.size] = xp.abs(R)

    metadata = {'frame_limit': frame_limit}

    if normalize:
        metadata.update(standard_name='Cyclic Autocorrelation')
    else:
        metadata.update(standard_name='Cyclic Autocovariance', units='mW')

    return result, metadata


@nb.njit(
    nb.void(
        nb.complex64[:], nb.int_[:], nb.int_, nb.complex64[:], nb.complex64[:], nb.complex64[:], nb.float32[:]
    ),
    parallel=True,
)
def _numba_indexed_cp_product(x, inds, fft_size, a_out, b_out, prod_out, power_out):
    """accelerated evaluation of the inner cyclic prefix indexing loop"""

    for i in nb.prange(inds.size):
        a_out[i] = x[inds[i] + fft_size]
        b_out[i] = np.conj(x[inds[i]])
        power_out[i] = 0.5 * (np.abs(a_out[i]) ** 2 + np.abs(b_out[i]) ** 2)
        prod_out[i] = a_out[i] * b_out[i]


def _indexed_cp_product(iq, cp_inds, fft_size):
    """evaluates the index-and-dot-product inner loop needed for
    correlating the cyclic prefixes in iq.
    """

    out = dict(
        a_out = np.empty(cp_inds.size, dtype=np.complex64),
        b_out = np.empty(cp_inds.size, dtype=np.complex64),
        prod_out = np.empty(cp_inds.size, dtype=np.complex64),
        power_out = np.empty(cp_inds.size, dtype=np.float32)
    )

    # numba accepts only flat arrays. flatten and then unflatten after exec
    _numba_indexed_cp_product(iq, cp_inds.flatten(), fft_size, **out)
    return [a.reshape(cp_inds.shape) for a in list(out.values())]


# %% TODO: the following low level code may be merged with similar functions in iqwaveform in the future
def _correlate_along_axis(a, b, axes=0, norm=False, _ab_product=None):
    """correlate `a` and `b` along the specified axes.

    if `norm`, normalize by the product of the standard deviates of
    a and b across the axes.

    if _product is passed in, it is assumed to contain the pre-computed
    element-wise product a*np.conj(b).
    """

    xp = iqwaveform.util.array_namespace(a)
    R = xp.sum(a * xp.conj(b), axis=axes)
    # if _ab_product is None:
    #     R = xp.sum(a * xp.conj(b), axis=axes)
    # else:
    #     R = xp.sum(_ab_product, axis=axes)

    if norm:
        R /= xp.std(a, axis=axes) * xp.std(b, axis=axes)

    return R


@lru_cache
def _get_phy_mappings(
    channel_bandwidth: float, subcarrier_spacings: tuple[float, ...], xp=np
) -> dict[str]:
    return {
        scs: ofdm.Phy3GPP(channel_bandwidth, scs, xp=xp) for scs in subcarrier_spacings
    }


@lru_cache
def _get_correlation_length(
    capture: structs.Capture, *, subcarrier_spacings: tuple[float, ...]
):
    phy_scs = _get_phy_mappings(capture.analysis_bandwidth, subcarrier_spacings)
    return max([np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()])


def _correlate_cyclic_prefixes(
    iq,
    phy,
    *,
    frames: tuple = (0,),
    sample_rate_error: float = 0.0,
    norm: bool = False,
    idx_reduction: str = 'corr',
    **idx_kwargs,
):
    """perform correlation at the all CP index offsets as used in maximum likelihood detection.

    Args:
        iq (_type_): _description_
        phy (PhyOFDM): _description_
        frames (tuple, optional): _description_. Defaults to (0,).
        sample_rate_error (float, optional): _description_. Defaults to 0.0.
        norm (bool, optional): _description_. Defaults to False.
        idx_reduction: how to reduce additional indexing axes:
            'corr' to correlate along the index, or None to return the np.ndarray for analysis
        idx_kwargs: keyword arguments passed to phy.index_cyclic_prefix

    Returns:
        _type_: _description_
    """
    xp = iqwaveform.util.array_namespace(iq)

    frames = iqwaveform.ofdm._index_or_all(
        frames,
        '"frames" argument',
        size=round(iq.size / phy.frame_size),
    )

    cp_inds = phy.index_cyclic_prefix(frames=tuple(frames), **idx_kwargs)

    if sample_rate_error != 0:
        cp_inds = (cp_inds * (1 + sample_rate_error)).round().astype(int)

    if idx_reduction == 'corr':
        corr_axes = tuple(range(cp_inds.ndim - 1))
    elif idx_reduction in ('peak', None):
        corr_axes = (-3, -2)
    else:
        raise ValueError('idx_reduction must be one of "corr", "peak", or None')

    # a, b, ab, power = _indexed_cp_product(iq, cp_inds, phy.fft_size)
    a = iq[cp_inds]
    b = iq[phy.nfft :][cp_inds]
    power = iqwaveform.envtopow(a)
    power += iqwaveform.envtopow(b)
    power /= 2

    R = -_correlate_along_axis(a, b, axes=corr_axes, norm=norm)  # , _ab_product=ab)

    corr_size = np.prod([cp_inds.shape[i] for i in corr_axes])
    R /= corr_size

    power = xp.mean(power, axis=corr_axes) / cp_inds.shape[-1]

    return R, power
