from __future__ import annotations

from ..dataarrays import expose_in_yaml, ChannelAnalysisResult, select_parameter_kws
from ..structs import Capture

from typing import Literal, get_args
from functools import lru_cache

from dataclasses import dataclass
import numpy as np

from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform

ofdm = iqwaveform.util.lazy_import('iqwaveform.ofdm')
pd = iqwaveform.util.lazy_import('pandas')


def correlate_along_axis(a, b, axes=0, norm=False, _ab_product=None):
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
def get_phy_mappings(channel_bandwidth: float, subcarrier_spacings: tuple[float, ...], xp=np) -> dict[str]:
    return {
        scs: ofdm.Phy3GPP(channel_bandwidth, scs, xp=xp)
        for scs in subcarrier_spacings
    }


@lru_cache
def get_correlation_length(capture: Capture, *, subcarrier_spacings: tuple[float, ...]):
    phy_scs = get_phy_mappings(capture.analysis_bandwidth, subcarrier_spacings)
    return max([np.diff(phy.cp_start_idx).min() for phy in phy_scs.values()])


def correlate_cyclic_prefixes(
    iq,
    phy,
    *,
    frames: tuple=(0,),
    sample_rate_error: float=0.0,
    norm: bool=False,
    idx_reduction: str='corr',
    **idx_kwargs
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

    cp_inds = phy.index_cyclic_prefix(
        frames=tuple(frames),
        **idx_kwargs
    )

    if sample_rate_error != 0:
        cp_inds = (cp_inds * (1 + sample_rate_error)).round().astype(int)

    if idx_reduction == 'corr':
        corr_axes = tuple(range(cp_inds.ndim-1))
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

    R = -correlate_along_axis(a, b, axes=corr_axes, norm=norm)#, _ab_product=ab)

    corr_size = np.prod([cp_inds.shape[i] for i in corr_axes])
    R /= corr_size

    power = xp.mean(power, axis=corr_axes) / cp_inds.shape[-1]

    return R, power


### iqwaveform wrapper
@expose_in_yaml
def cellular_cyclic_autocorrelation(
    iq,
    capture: Capture,
    *,
    subcarrier_spacings: tuple[float, ...] = (15e3, 30e3, 60e3),
) -> ChannelAnalysisResult:
    params = select_parameter_kws(locals())

    xp = iqwaveform.util.array_namespace(iq)
    subcarrier_spacings = tuple(subcarrier_spacings)
    phy_scs = get_phy_mappings(capture.analysis_bandwidth, subcarrier_spacings, xp=xp)

    kws = dict(slots='all', symbols='all', frames=np.arange(0,10), norm=True)

    max_len = get_correlation_length(capture, subcarrier_spacings=subcarrier_spacings)

    result = xp.full((len(subcarrier_spacings), max_len), np.nan, dtype=np.float32)
    for i, phy in enumerate(phy_scs.values()):
        R, _ = correlate_cyclic_prefixes(iq, phy, **kws)
        result[i][:R.size] = xp.abs(R)

    metadata = {
        'standard_name': 'Correlation',
    }

    return ChannelAnalysisResult(
        CellularCyclicAutocorrelation, xp.array(result), capture, parameters=params, attrs=metadata
    )


### Time elapsed dimension and coordinates
CyclicSampleLagAxis = Literal['cyclic_sample_lag']

@lru_cache
def cyclic_sample_lag_coord_factory(
    capture: Capture, *, subcarrier_spacings: tuple[float, ...], **_
) -> dict[str, np.ndarray]:
    
    max_len = get_correlation_length(capture, subcarrier_spacings=subcarrier_spacings)
    return pd.RangeIndex(0, max_len, name=get_args(CyclicSampleLagAxis)[0])


@dataclass
class CyclicSampleLagCoords:
    data: Data[CyclicSampleLagAxis, np.float32]
    standard_name: Attr[str] = 'Cyclic symbol index offset'
    # units: Attr[str] = 's'

    factory: callable = cyclic_sample_lag_coord_factory


### Subcarrier spacing label axis
SubcarrierSpacingAxis = Literal['subcarrier_spacing']


@lru_cache
def subcarrier_spacing_coord_factory(capture: Capture, *, subcarrier_spacings: tuple[float, ...], **_):
    return tuple(subcarrier_spacings)


@dataclass
class SubcarrierSpacingCoords:
    data: Data[SubcarrierSpacingAxis, np.float32]
    standard_name: Attr[str] = 'Cellular subcarrier spacing'
    units: Attr[str] = 'Hz'

    factory: callable = subcarrier_spacing_coord_factory


### Dataarray definition
@dataclass
class CellularCyclicAutocorrelation(AsDataArray):
    power_time_series: Data[tuple[SubcarrierSpacingAxis,CyclicSampleLagAxis], np.float32]

    subcarrier_spacing: Coordof[SubcarrierSpacingCoords]
    cyclic_sample_lag: Coordof[CyclicSampleLagCoords]

    standard_name: Attr[str] = 'Correlation'
    # units: Attr[str] = 'dBm'
