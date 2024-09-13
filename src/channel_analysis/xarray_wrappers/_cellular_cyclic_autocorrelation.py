from __future__ import annotations
import typing
from functools import lru_cache
import numpy as np
import numba as nb
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Coordof, Data, Attr
import iqwaveform
from .. import structs, dataarrays
import array_api_compat
import math

ofdm = iqwaveform.util.lazy_import('iqwaveform.ofdm')
pd = iqwaveform.util.lazy_import('pandas')


### Time elapsed dimension and coordinates
CyclicSampleLagAxis = typing.Literal['cyclic_sample_lag']


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
        axis_name = typing.get_args(CyclicSampleLagAxis)[0]
        return pd.RangeIndex(0, max_len, name=axis_name)


### Subcarrier spacing label axis
SubcarrierSpacingAxis = typing.Literal['subcarrier_spacing']


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
    frame_range: tuple[int,typing.Optional[int]] = (0, 1),
    slot_range: tuple[int,typing.Optional[int]] = (0, None),
    symbol_range: tuple[int,typing.Optional[int]] = (0, None),
    normalize: bool = True,
):
    RANGE_MAP = {'frames': frame_range, 'slots': slot_range, 'symbols': symbol_range}
    xp = iqwaveform.util.array_namespace(iq)
    subcarrier_spacings = tuple(subcarrier_spacings)
    phy_scs = _get_phy_mappings(capture.analysis_bandwidth, subcarrier_spacings, xp=xp)
    metadata = {}

    kws = {'norm': normalize}
    for name, field_range in RANGE_MAP.items():
        field_range = tuple(field_range)
        metadata[name] = field_range

        if field_range in ((0,), (None, None), (0,None)):
            kws[name] = 'all'
        else:
            kws[name] = tuple(range(*field_range))

    max_len = _get_correlation_length(capture, subcarrier_spacings=subcarrier_spacings)

    result = xp.full((len(subcarrier_spacings), max_len), np.nan, dtype=np.float32)
    for i, phy in enumerate(phy_scs.values()):
        R = _correlate_cyclic_prefixes(iq, phy, **kws)
        result[i][: R.size] = xp.abs(R)

    if normalize:
        metadata.update(standard_name='Cyclic Autocorrelation')
    else:
        metadata.update(standard_name='Cyclic Autocovariance', units='mW')

    return result, metadata


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

    # if idx_reduction == 'corr':
    #     corr_axes = tuple(range(cp_inds.ndim - 1))
    # elif idx_reduction in ('peak', None):
    #     corr_axes = (-3, -2)
    # else:
    #     raise ValueError('idx_reduction must be one of "corr", "peak", or None')

    # R = -_correlate_along_axis(a, b, axes=corr_axes, norm=norm, _summand=summand)
    R = corr_at_indices(cp_inds, iq, phy.nfft, norm=norm)

    # corr_size = np.prod([cp_inds.shape[i] for i in corr_axes])
    # R /= corr_size

    # power = xp.mean(power, axis=corr_axes) / cp_inds.shape[-1]

    return R#, power


try:
    from ..cuda_kernels import _corr_at_indices_cuda
except ModuleNotFoundError:
    pass


@nb.njit(
    [
        (nb.int32[:,:], nb.complex64[:], nb.int32, nb.boolean, nb.complex64[:]),
        (nb.int32[:,:], nb.complex64[:], nb.int64, nb.boolean, nb.complex64[:]),
        (nb.int64[:,:], nb.complex64[:], nb.int32, nb.boolean, nb.complex64[:]),
        (nb.int64[:,:], nb.complex64[:], nb.int64, nb.boolean, nb.complex64[:]),
    ],
    parallel=True
)
def _corr_at_indices_cpu(inds, x, nfft, norm, out):
    for j in nb.prange(inds.shape[1]):
        accum_corr = nb.complex128(0+0j)
        accum_power_a = nb.float64(0.0)
        accum_power_b = nb.float64(0.0)
        for i in range(inds.shape[0]):
            ix = inds[i,j]
            a = x[ix]
            b = x[ix+nfft].conjugate()
            accum_corr += a*b
            if norm:
                accum_power_a += a.real*a.real+a.imag*a.imag
                accum_power_b += b.real*b.real+b.imag*b.imag

        if norm:
            # normalized by the standard deviation, assuming zero-mean 
            accum_corr /= math.sqrt(accum_power_a*accum_power_b)/inds.shape[0]

        out[j] = accum_corr


def corr_at_indices(inds, x, nfft, norm=True, out=None):
    xp = iqwaveform.util.array_namespace(x)

    if out is None:
        out = xp.empty(inds.shape[-1], dtype=x.dtype)

    if inds.ndim > 2:
        new_shape = np.prod(inds.shape[:-1]), inds.shape[-1]
        inds = inds.reshape(new_shape)

    if xp is array_api_compat.numpy:
        _corr_at_indices_cpu(inds, x, nfft, norm, out)

    else:
        tpb = 32
        bpg = (x.size + (tpb - 1)) // tpb

        _corr_at_indices_cuda[bpg,tpb](inds, x, np.int32(nfft), norm, out)

    out /= inds.shape[0]

    return out
