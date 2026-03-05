from __future__ import annotations as __

import dataclasses
import typing
from fractions import Fraction
from math import ceil

from .. import specs

from ..lib import util
from . import _channel_power_histogram, _spectrogram, shared
from ._cellular_cyclic_autocorrelation import link_direction, tdd_config_from_str
from .shared import registry, hint_keywords

import striqt.waveform as sw

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


@dataclasses.dataclass
class LinkPair:
    downlink: typing.Any
    uplink: typing.Any


@registry.coordinates(
    dtype='float32',
    attrs={'standard_name': 'Cellular resource grid bin power', 'units': 'dBm'},
)
@util.lru_cache()
def cellular_resource_power_bin(
    capture: specs.Capture, spec: specs.CellularResourcePowerHistogram
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """returns a dictionary of coordinate values, keyed by axis dimension name"""

    bins = _channel_power_histogram.make_power_bins(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
    )

    fres = spec.subcarrier_spacing / 2

    if sw.isroundmod(capture.sample_rate, fres):
        # need capture.sample_rate/resolution to give us a counting number
        nfft = round(capture.sample_rate / fres)
    else:
        raise ValueError('sample_rate/resolution must be a counting number')

    integration_bandwidth = _get_integration_bandwidth(spec)
    metadata = {
        'noise_bandwidth': float(integration_bandwidth),
        'units': f'dBm/{integration_bandwidth / 1e3:0.0f} kHz',
    }
    return bins, metadata


def apply_mask(
    spectrogram,
    freqs,
    *,
    channel_bandwidth,
    subcarrier_spacing,
    frame_slots: str,
    special_symbols: typing.Optional[str],
    guard_left=None,
    guard_right=None,
    link_direction=('downlink', 'uplink'),
    flex_as=None,
    normal_cp=True,
    xp=np,
) -> util.ArrayType:
    """splits the spectrogram into TDD downlink and uplink components that are masked
    with `float('nan')`.

    See also:
        `build_tdd_link_symbol_masks`
    """

    if isinstance(link_direction, str):
        # ensure link_direction is a tuple
        link_direction = (link_direction,)

    if (
        len(link_direction)
        - link_direction.count('downlink')
        - link_direction.count('uplink')
        > 0
    ):
        raise ValueError(
            'only "downlink" or "uplink" are valid values for link_direction tuple'
        )

    # null frequencies in the guard interval
    eps = 1e-6
    ilo = xp.searchsorted(freqs, xp.asarray(-channel_bandwidth / 2 + guard_left + eps))
    ihi = xp.searchsorted(freqs, xp.asarray(channel_bandwidth / 2 - guard_right - eps))
    spg_left = sw.util.axis_slice(spectrogram, 0, ilo, axis=-1)
    spg_right = sw.util.axis_slice(spectrogram, ihi, None, axis=-1)
    xp.copyto(spg_left, float('nan'))
    xp.copyto(spg_right, float('nan'))

    # null in time to select each of the {down,up} links
    masks = build_tdd_link_symbol_masks(
        subcarrier_spacing=subcarrier_spacing,
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        link_direction=link_direction,
        count=spectrogram.shape[-2],
        xp=xp,
        flex_as=flex_as,
        normal_cp=normal_cp,
    )

    # broadcast into dimensions (input channel, link direction, symbols elapsed, frequency)
    return masks[np.newaxis, :, :, np.newaxis] * spectrogram[:, np.newaxis, :, :]


@util.lru_cache()
def build_tdd_link_symbol_masks(
    subcarrier_spacing: float,
    frame_slots: str,
    special_symbols: typing.Optional[str] = None,
    *,
    link_direction: tuple[str, ...] = ('downlink', 'uplink'),
    count: int | None = None,
    normal_cp=True,
    flex_as=None,
    xp=np,
) -> util.ArrayType:
    """generate a symbol-by-symbol sequence of masking arrays for uplink and downlink.

    The number of slots given in the frame match the appropriate number for a given
    5G NR or LTE subcarrier spacing.

    Arguments:
        frame_slots: a string composed of the characters {'d', 'u', 's'} that
            indicate the sequence of slots in 1 cellular frame
        special_symbols: the a string composed of the characters {'d', 'u', 'f'} that
            indicate the sequence of symbol types in the special slot.
    """

    tdd_config = tdd_config_from_str(
        subcarrier_spacing=subcarrier_spacing,
        frame_slots=frame_slots,
        special_symbols=special_symbols,
        normal_cp=normal_cp,
        flex_as=flex_as,
    )

    out_shape = (len(link_direction), count)
    out = xp.empty(out_shape, dtype='float32')
    for i, direction in enumerate(link_direction):
        single_mask = [
            tdd_config.code_maps[direction][k] for k in tdd_config.frame_by_symbol
        ]

        if count is None:
            frame_count = 1
        else:
            frame_count = ceil(count / len(single_mask))

        mask = (single_mask * frame_count)[:count]
        out[i] = xp.asarray(mask)

    return out


def _get_integration_bandwidth(
    spec: specs.CellularResourcePowerHistogram,
) -> float:
    if spec.average_rbs == 'half':
        return 6 * spec.subcarrier_spacing
    elif spec.average_rbs:
        return 12 * spec.subcarrier_spacing
    else:
        return spec.subcarrier_spacing


def _struct_defaults(spec_type: type[specs.SpecBase]) -> dict[str, typing.Any]:
    defaults = spec_type.__struct_defaults__
    fields = spec_type.__struct_fields__

    # defaults specify only the end of the fields list
    return dict(zip(fields[-len(defaults) :], defaults))


@hint_keywords(specs.CellularResourcePowerHistogram)
@registry.measurement(
    coord_factories=[link_direction, cellular_resource_power_bin],
    dtype='float32',
    depends=_spectrogram.spectrogram,
    spec_type=specs.CellularResourcePowerHistogram,
    attrs={'standard_name': 'Fraction of resource grid'},
)
def cellular_resource_power_histogram(
    iq: 'sw.util.ArrayLike',
    capture: specs.Capture,
    **kwargs,
):
    """

    Args:
        window: window function to use (matching the arguments to scipy.signal.get_window)
        subcarrier_spacing (Hz): 15e3|30e3|60e3|120e3|240e3|480e3|960e3
        power_low (dB arb units): bottom edge of the histogram bins
        power_high (dB arb units): top edge of the histogram bins
        power_resolution (dB): resolution of the histogram
        frame_slots: string composed of {'d', 'u', 's'} that speficy the sequence
            of slots in 1 TDD cellular frame, or None to fill with downlink.
        special_symbols: string composed of {'d', 'u', 'f'} that
            indicate the sequence of symbol types (when 's' is in frame_slots).
        guard_bandwidths (in Hz): the channel guard bandwidths on the left and right sides
        average_rbs: when True, counts will report power averaged to 1 RB in frequency;
            if False, resolution is 1 subcarrier.
        average_slot: when True, counts will report power averaged to 1 slot in time;
            if False, resolution is 1 symbol.
        cyclic_prefix: the 3GPP cyclic prefix size, one of
            `('extended','normal')`
        lo_bandstop: if specified, null this bandwidth about DC
        as_xarray: if True (the default), returns an xarray with labeled axes and metadata;
            otherwise, returns an (array, dict) tuple containing the result and metadata

    Returns:
        `xarray.DataArray` or `(array, dict)` based on `as_xarray`
    """
    spec = specs.CellularResourcePowerHistogram.from_dict(kwargs)
    spec_defaults = _struct_defaults(specs.CellularResourcePowerHistogram)

    xp = sw.util.array_namespace(iq)

    link_direction = 'downlink', 'uplink'

    integration_bandwidth = _get_integration_bandwidth(spec)

    if spec.average_slots:
        time_aperture = 1e-3 * (15e3 / spec.subcarrier_spacing)
    else:
        time_aperture = None

    slot_count = round(10 * spec.subcarrier_spacing / 15e3)
    frame_slots = spec.frame_slots
    if frame_slots is None:
        frame_slots = slot_count * 'd'
    elif len(frame_slots) != slot_count:
        raise ValueError(
            f'expected a string with {slot_count} characters, but received {len(frame_slots)}'
        )

    if 's' in frame_slots and spec.special_symbols is None:
        raise ValueError(
            'specify special_symbols that implement the requested "s" special slot'
        )

    # set STFT overlap and the fractional fill in the window
    if spec.cyclic_prefix == 'normal':
        fractional_overlap = Fraction(13, 28)
        window_fill = Fraction(15, 28)
    elif spec.cyclic_prefix == 'extended':
        fractional_overlap = Fraction(11, 24)
        window_fill = Fraction(13, 24)
    else:
        raise ValueError('cp_guard_period must be "normal" or "extended"')

    spg_spec = specs.Spectrogram(
        window=spec.window,
        frequency_resolution=spec.subcarrier_spacing / 2,
        fractional_overlap=fractional_overlap,
        window_fill=window_fill,
        integration_bandwidth=integration_bandwidth,
        lo_bandstop=spec.lo_bandstop,
    )

    if time_aperture is None:
        time_bin_averaging = None
    else:
        nfft = capture.sample_rate / spg_spec.frequency_resolution
        noverlap = fractional_overlap * nfft
        hop_size = nfft - noverlap
        hop_period = hop_size / capture.sample_rate
        time_bin_averaging = round(time_aperture / hop_period)

        assert sw.isroundmod(time_aperture / hop_period, 1)

    spg, metadata = shared.evaluate_spectrogram(
        iq, capture, spg_spec, dtype='float32', dB=False
    )
    del metadata['units']

    freqs = shared.spectrogram_baseband_frequency(capture, spg_spec, xp=xp)

    masked_spgs = apply_mask(
        spg,
        freqs,
        subcarrier_spacing=spec.subcarrier_spacing,
        link_direction=link_direction,
        channel_bandwidth=capture.analysis_bandwidth,
        frame_slots=frame_slots,
        special_symbols=spec.special_symbols,
        guard_left=spec.guard_bandwidths[0],
        guard_right=spec.guard_bandwidths[1],
        xp=xp,
    )

    # apply the time binning only now, to allow for averaging
    # across mask boundaries
    if time_bin_averaging is not None:
        masked_spgs = sw.util.binned_mean(
            masked_spgs, time_bin_averaging, axis=2, fft=False
        )

    masked_spgs = sw.powtodB(masked_spgs, out=masked_spgs)
    bin_edges = _channel_power_histogram.make_power_histogram_bin_edges(
        power_low=spec.power_low,
        power_high=spec.power_high,
        power_resolution=spec.power_resolution,
        xp=xp,
    )

    flat_shape = masked_spgs.shape[:2] + (-1,)
    counts, _ = sw.histogram_last_axis(masked_spgs.reshape(flat_shape), bin_edges)

    norm = xp.sum(counts, axis=(1, 2), keepdims=True)
    norm[norm == 0] = 1
    data = counts / norm

    return data, metadata
