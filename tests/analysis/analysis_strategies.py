"""strategies and oracles for the analysis tests: the capture/delay/range domains
that __post_init__ validates, the (capture, spec) domains of every registered
validator, the freeze/unfreeze and frozendict trees, and the registry walk that
fetches a measurement's tolerance.

Not a conftest: a second rootless conftest would shadow the root one that other test
directories import by name.
"""

from __future__ import annotations

from fractions import Fraction
from math import inf

from conftest import scalars
from hypothesis import strategies as st

import striqt.analysis as sa
from striqt.analysis.specs.helpers import frozendict

# %% registered tolerances


def registered_tolerance(capture, spec, **kwargs) -> sa.specs.Tolerance:
    """the tolerance registered for the measurement that `spec` selects, fetched
    through the registry walk a consumer would use"""
    name = sa.registry[type(spec)].name
    group = sa.registry.tospec()(**{name: spec})
    return sa.registry.tolerances(capture, group, **kwargs)[name]


# %% capture, delay and range domains

_SAMPLE_RATES = (1e6, 2e6, 7.68e6, 15.36e6)
_SSB_SAMPLE_RATE = 7.68e6

# a fractional part of at least 1e-3 samples stays clear of the 1e-6 sample
# tolerance in Capture.__post_init__
fractional_parts = st.floats(min_value=1e-3, max_value=1 - 1e-3)


@st.composite
def integer_sample_captures(draw):
    sample_rate = draw(st.sampled_from(_SAMPLE_RATES))
    count = draw(st.integers(min_value=1, max_value=10**6))
    return {'duration': count / sample_rate, 'sample_rate': sample_rate}


@st.composite
def fractional_sample_captures(draw):
    sample_rate = draw(st.sampled_from(_SAMPLE_RATES))
    count = draw(st.integers(min_value=0, max_value=10**6))
    frac = draw(fractional_parts)
    return {'duration': (count + frac) / sample_rate, 'sample_rate': sample_rate}


@st.composite
def integer_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    return {'sample_rate': _SSB_SAMPLE_RATE, 'delay': n / _SSB_SAMPLE_RATE}


@st.composite
def fractional_delay_kwargs(draw):
    n = draw(st.integers(min_value=0, max_value=10**5))
    frac = draw(fractional_parts)
    return {'sample_rate': _SSB_SAMPLE_RATE, 'delay': (n + frac) / _SSB_SAMPLE_RATE}


range_starts = st.integers(min_value=0, max_value=50)


@st.composite
def valid_frame_ranges(draw):
    start = draw(range_starts)
    if start > 0:
        stop = draw(st.integers(min_value=start, max_value=60))
    else:
        stop = draw(st.integers(min_value=-5, max_value=60))
    return (start, stop)


@st.composite
def valid_symbol_ranges(draw):
    start = draw(range_starts)
    if start == 0:
        stop = draw(st.one_of(st.none(), st.integers(min_value=-5, max_value=60)))
    else:
        stop = draw(st.integers(min_value=start, max_value=60))
    return (start, stop)


@st.composite
def descending_ranges(draw):
    start = draw(st.integers(min_value=1, max_value=50))
    stop = draw(st.integers(min_value=-5, max_value=start - 1))
    return (start, stop)


# %% (capture, spec) domains of the registered validators
#
# Each strategy draws pairs that straddle a validator's rules, so that the pass and
# reject regions both get exercised: sample counts below one FFT and odd, offsets off
# the resampler grid, durations that are not whole 10 ms frames, statistic names and
# quantiles outside [0, 1], frame ranges past the capture, analysis bandwidths above
# the sample rate. The power bins are fixed: no validator reads them.

POWER_BINS = {'power_low': -40.0, 'power_high': 10.0, 'power_resolution': 1.0}

STATISTICS = ('mean', 'max', 'min', 'rms', 'peak', 'median', 0.5, 0.25)
BAD_STATISTICS = (1.5, -0.1, 'bogus')
# power_detectors is typed tuple[str, ...], so a quantile never reaches its validator
DETECTORS = ('rms', 'peak', 'max', 'min', 'mean', 'median')

# the sample rates at which the 3GPP cyclic prefixes are whole samples: multiples of
# 1.92e6 at 15 kHz and of 3.84e6 at 30 kHz (3GPP TS 38.211 Section 5.3.1)
CELL_FS = 3.84e6
TDD_20_SLOTS = 'dddsuudddddddsuudddd'
TDD_10_SLOTS = 'dsuuuuuuuu'


def _capture(samples: int, sample_rate: float, analysis_bandwidth=inf):
    return sa.specs.Capture(
        duration=samples / sample_rate,
        sample_rate=sample_rate,
        analysis_bandwidth=analysis_bandwidth,
    )


def _split_capture(params: dict) -> tuple[sa.specs.Capture, dict]:
    """pop the capture fields off a drawn parameter set"""
    params = dict(params)
    capture = _capture(
        params.pop('samples'),
        params.pop('sample_rate'),
        params.pop('analysis_bandwidth', inf),
    )
    return capture, params


def _statistic_tuples(values, max_size=3):
    return st.lists(st.sampled_from(values), min_size=1, max_size=max_size).map(tuple)


def _perturbed(draw, valid: dict, perturbations: dict) -> dict:
    """`valid`, with one field replaced by a drawn failure-mode value half the time"""
    if not draw(st.booleans()):
        return valid
    field = draw(st.sampled_from(sorted(perturbations)))
    return {**valid, field: draw(perturbations[field])}


@st.composite
def stft_pairs(draw, spec_type=sa.specs.Spectrogram, **extra):
    """(capture, spec) for the spectrogram family, with `extra` fields drawn as given"""
    fs = draw(st.sampled_from((1.024e6, 2.048e6)))
    nfft = draw(st.sampled_from((8, 16, 64)))
    resolution = fs / nfft
    fractional_overlap = draw(st.sampled_from((Fraction(0), Fraction(1, 2))))
    hop_period = float(1 - fractional_overlap) * nfft / fs
    valid = {
        'samples': draw(st.sampled_from((64, 128, 1024))),
        'sample_rate': fs,
        'analysis_bandwidth': draw(st.sampled_from((inf, fs / 2, fs))),
        'window': draw(st.sampled_from(('boxcar', 'hann', 'hamming', ('kaiser', 3.0)))),
        'frequency_resolution': resolution,
        'fractional_overlap': fractional_overlap,
        'window_fill': draw(st.sampled_from((Fraction(1), Fraction(1, 2)))),
        'integration_bandwidth': draw(st.sampled_from((None, 2 * resolution))),
        'trim_stopband': draw(st.booleans()),
        'lo_bandstop': draw(st.sampled_from((None, resolution, 2 * resolution))),
    }
    perturbations = {
        'samples': st.sampled_from((3, 17, 63)),
        'analysis_bandwidth': st.sampled_from((1.5 * fs, fs / 3)),
        'frequency_resolution': st.sampled_from((1.5 * resolution, fs / 7, fs / 9)),
        'integration_bandwidth': st.sampled_from((1.5 * resolution, 2 * fs)),
        'lo_bandstop': st.sampled_from((1.5 * resolution, 2 * fs)),
    }
    if 'time_aperture' in spec_type.__struct_fields__:
        valid['time_aperture'] = draw(st.sampled_from((None, hop_period)))
        perturbations['time_aperture'] = st.sampled_from((
            2 * hop_period,
            1.5 * hop_period,
        ))
    for name, (strategy, bad) in extra.items():
        valid[name] = draw(strategy)
        if bad is not None:
            perturbations[name] = bad
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    return capture, spec_type(**kwargs)


@st.composite
def detector_pairs(draw, spec_type=sa.specs.ChannelPowerTimeSeries, **extra):
    """(capture, spec) for the binned power detectors, with `extra` fields drawn"""
    valid = {
        'samples': draw(st.sampled_from((100, 1000))),
        'sample_rate': 1e6,
        'detector_period': draw(
            st.sampled_from((Fraction(1, 100_000), Fraction(1, 50_000)))
        ),
        'power_detectors': draw(_statistic_tuples(DETECTORS)),
    }
    perturbations = {
        'samples': st.sampled_from((30, 105)),
        'detector_period': st.sampled_from((Fraction(1, 300_000), Fraction(1, 1000))),
        'power_detectors': st.just(('rms', 'bogus')),
    }
    for name, (strategy, bad) in extra.items():
        valid[name] = draw(strategy)
        if bad is not None:
            perturbations[name] = bad
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    return capture, spec_type(**kwargs)


@st.composite
def ssb_correlator_pairs(draw, spec_type=sa.specs.Cellular5GNRPSSCorrelator, **extra):
    """(capture, spec) for the PSS/SSS correlators and synchronizers"""
    frame_size = round(10e-3 * CELL_FS)
    samples = draw(st.integers(min_value=1, max_value=2)) * frame_size
    step = CELL_FS / samples
    scs = draw(st.sampled_from((15e3, 30e3)))
    if round(scs) == 15_000:
        symbol_indexes = st.sampled_from(('auto', 'a'))
        # the correlator rate must be a multiple of 128 subcarrier spacings
        sample_rate_out = draw(st.sampled_from((CELL_FS, CELL_FS / 2)))
        bad_rates = (5e6,)
    else:
        # 'auto' has no case at 30 kHz outside shared spectrum
        symbol_indexes = st.sampled_from(('c', (2, 8)))
        sample_rate_out = CELL_FS
        bad_rates = (CELL_FS / 2, 5e6)
    # `resample` shifts only while downsampling
    offsets = (0.0,) if sample_rate_out == CELL_FS else (0.0, 100 * step)
    valid = {
        'samples': samples,
        'sample_rate': CELL_FS,
        'subcarrier_spacing': scs,
        'sample_rate_out': sample_rate_out,
        'discovery_periodicity': draw(st.sampled_from((10e-3, 20e-3))),
        'frequency_offset': draw(st.sampled_from(offsets)),
        'shared_spectrum': False,
        'symbol_indexes': draw(symbol_indexes),
        'max_lag_symbols': draw(st.sampled_from((None, 1, 4, 6))),
    }
    perturbations = {
        'samples': st.sampled_from((samples + 1, samples + frame_size // 2)),
        'sample_rate': st.just(1.92e6),
        'sample_rate_out': st.sampled_from(bad_rates),
        'discovery_periodicity': st.just(15e-3),
        'frequency_offset': st.sampled_from((
            100.5 * step,
            CELL_FS / 4,
            -CELL_FS / 4,
            CELL_FS / 2,
        )),
        'shared_spectrum': st.just(True),
        'symbol_indexes': st.sampled_from(('auto', 'b')),
    }
    for name, (strategy, bad) in extra.items():
        valid[name] = draw(strategy)
        if bad is not None:
            perturbations[name] = bad
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    kwargs['sample_rate'] = kwargs.pop('sample_rate_out')
    return capture, spec_type(**kwargs)


@st.composite
def ssb_spectrogram_pairs(draw):
    fs = draw(st.sampled_from((420e3, 840e3)))
    scs = draw(st.sampled_from((15e3, 30e3)))
    period = draw(st.sampled_from((10e-3, 20e-3)))
    # the burst set is the first 2 of the 10*scs/15e3 slots in a 10 ms frame, and
    # the STFT yields its last symbol only once a whole window (2/scs) follows
    burst = 2e-3 * 15e3 / scs
    trailing_burst = period + 2 * burst + 2 / scs
    valid = {
        'samples': round(
            draw(st.sampled_from((period, 2 * period, trailing_burst))) * fs
        ),
        'sample_rate': fs,
        'subcarrier_spacing': scs,
        'sample_rate_out': draw(st.sampled_from((fs / 2, 4 * scs))),
        'discovery_periodicity': period,
        'frequency_offset': draw(st.sampled_from((0.0, scs, -2 * scs))),
        'max_block_count': draw(st.sampled_from((None, 1))),
        'window': draw(st.sampled_from(('boxcar', 'blackmanharris'))),
        'lo_bandstop': draw(st.sampled_from((None, scs))),
    }
    perturbations = {
        'samples': st.sampled_from((
            round(burst * fs),
            round((period + burst / 2) * fs),
        )),
        'sample_rate': st.just(300e3),
        'sample_rate_out': st.just(2 * fs),
        'frequency_offset': st.sampled_from((scs / 2, fs)),
        'lo_bandstop': st.just(2 * fs),
    }
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    kwargs['sample_rate'] = kwargs.pop('sample_rate_out')
    return capture, sa.specs.Cellular5GNRSSBSpectrogram(**kwargs)


@st.composite
def autocorrelation_pairs(draw):
    fs = draw(st.sampled_from((CELL_FS, 2 * CELL_FS)))
    frames = draw(st.integers(min_value=1, max_value=2))
    scs_choices = [15e3, 30e3, (15e3, 30e3)]
    if fs == 2 * CELL_FS:
        scs_choices += [60e3, (15e3, 30e3, 60e3)]
    scs = draw(st.sampled_from(scs_choices))
    frame_ranges = [(0, 1), 1] + ([(0, 2), (1, 2)] if frames == 2 else [])
    # one TDD pattern length fits every spacing only when there is one spacing
    frame_slots = [None, 'd']
    if isinstance(scs, float):
        frame_slots.append(TDD_20_SLOTS if round(scs) == 30_000 else TDD_10_SLOTS)
    valid = {
        'samples': frames * round(10e-3 * fs),
        'sample_rate': fs,
        'subcarrier_spacings': scs,
        'frame_range': draw(st.sampled_from(frame_ranges)),
        'frame_slots': draw(st.sampled_from(frame_slots)),
        'symbol_range': draw(
            st.sampled_from(((0, None), (0, 14), (2, 5), (13, 14), (-14, 0), 3))
        ),
        'generation': draw(st.sampled_from(('4G', '5G'))),
    }
    perturbations = {
        'samples': st.just(round(5e-3 * fs)),
        'subcarrier_spacings': st.just(60e3),
        'frame_range': st.sampled_from(((0, 3), (2, 3), (0, 0), 2)),
        'frame_slots': st.sampled_from(('u', 'd' * 39, TDD_10_SLOTS, TDD_20_SLOTS)),
        'symbol_range': st.sampled_from(((0, 15), (14, 15), (-15, 0), (0, 0))),
    }
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    return capture, sa.specs.CellularCyclicAutocorrelator(**kwargs)


@st.composite
def resource_grid_pairs(draw):
    fs = draw(st.sampled_from((420e3, 840e3)))
    scs = draw(st.sampled_from((15e3, 30e3)))
    tdd = TDD_20_SLOTS if round(scs) == 30_000 else TDD_10_SLOTS
    valid = {
        'samples': round(draw(st.sampled_from((1e-3, 2e-3, 5e-3))) * fs),
        'sample_rate': fs,
        'analysis_bandwidth': draw(st.sampled_from((inf, fs / 2, 0.7 * fs, fs))),
        'window': draw(st.sampled_from(('hamming', 'boxcar'))),
        'subcarrier_spacing': scs,
        'average_rbs': draw(st.sampled_from((False, True, 'half'))),
        'average_slots': draw(st.booleans()),
        'guard_bandwidths': draw(st.sampled_from(((0, 0), (15e3, 15e3)))),
        'frame_slots': draw(st.sampled_from((None, 'd', tdd))),
        'special_symbols': 'ddddddfffffffu',
        'cyclic_prefix': 'normal',
        'lo_bandstop': draw(st.sampled_from((None, 2 * scs))),
        **POWER_BINS,
    }
    perturbations = {
        'samples': st.just(round(0.25e-3 * fs)),
        'sample_rate': st.just(300e3),
        'analysis_bandwidth': st.just(1.5 * fs),
        'special_symbols': st.just(None),
        'cyclic_prefix': st.just('extended'),
        'lo_bandstop': st.just(2 * fs),
    }
    capture, kwargs = _split_capture(_perturbed(draw, valid, perturbations))
    return capture, sa.specs.CellularResourcePowerHistogram(**kwargs)


_FIXED_BINS = {k: (st.just(v), None) for k, v in POWER_BINS.items()}
_SYNC_FIELDS = {
    'window_fill': (st.sampled_from((1, 0.5)), None),
    'per_port': (st.booleans(), None),
    'max_beams': (st.sampled_from((None, 1, 2)), None),
}
_BLOCKS = {'max_block_count': (st.sampled_from((None, 1, 2)), None)}

VALIDATOR_DOMAINS = {
    'spectrogram': stft_pairs(),
    'power_spectral_density': stft_pairs(
        sa.specs.PowerSpectralDensity,
        time_statistic=(
            _statistic_tuples(STATISTICS),
            _statistic_tuples(BAD_STATISTICS, max_size=1),
        ),
    ),
    'spectrogram_histogram': stft_pairs(sa.specs.SpectrogramHistogram, **_FIXED_BINS),
    'spectrogram_ratio_histogram': stft_pairs(
        sa.specs.SpectrogramHistogramRatio, **_FIXED_BINS
    ),
    'channel_power_time_series': detector_pairs(),
    'channel_power_histogram': detector_pairs(
        sa.specs.ChannelPowerHistogram, **_FIXED_BINS
    ),
    'cyclic_channel_power': detector_pairs(
        sa.specs.CyclicChannelPower,
        cyclic_period=(st.just(1e-4), st.just(2.5e-5)),
        cyclic_statistics=(
            _statistic_tuples(STATISTICS),
            _statistic_tuples(BAD_STATISTICS, max_size=1),
        ),
    ),
    'cellular_5g_pss_correlation': ssb_correlator_pairs(
        sa.specs.Cellular5GNRPSSCorrelator, **_BLOCKS
    ),
    'cellular_5g_sss_correlation': ssb_correlator_pairs(
        sa.specs.Cellular5GNRSSSCorrelator, **_BLOCKS
    ),
    'cellular_5g_pss_sync': ssb_correlator_pairs(
        sa.specs.Cellular5GNPSSSync, **_SYNC_FIELDS
    ),
    'cellular_5g_sss_sync': ssb_correlator_pairs(
        sa.specs.Cellular5GNSSSSync, **_SYNC_FIELDS
    ),
    'cellular_5g_ssb_spectrogram': ssb_spectrogram_pairs(),
    'cellular_cyclic_autocorrelation': autocorrelation_pairs(),
    'cellular_resource_power_histogram': resource_grid_pairs(),
}


# %% freeze/unfreeze and frozendict


def _children(obj):
    if isinstance(obj, (dict, frozendict)):
        return obj.values()
    if isinstance(obj, (list, tuple)):
        return obj
    return ()


def container_depth(obj) -> int:
    if isinstance(obj, (list, tuple, dict, frozendict)):
        return 1 + max((container_depth(v) for v in _children(obj)), default=0)
    return 0


def has_mutable_below(obj, level: int = 0) -> bool:
    """True if a list or dict sits at nesting level >= level (the root is level 0)"""
    if level <= 0 and isinstance(obj, (list, dict)):
        return True
    return any(has_mutable_below(v, level - 1) for v in _children(obj))


def has_frozen_below(obj, level: int = 0) -> bool:
    """True if a tuple or frozendict sits at nesting level >= level"""
    if level <= 0 and isinstance(obj, (tuple, frozendict)):
        return True
    return any(has_frozen_below(v, level - 1) for v in _children(obj))


def tuplify(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(tuplify(v) for v in obj)
    if isinstance(obj, (dict, frozendict)):
        return frozendict({k: tuplify(v) for k, v in obj.items()})
    return obj


def listify(obj):
    if isinstance(obj, (list, tuple)):
        return [listify(v) for v in obj]
    if isinstance(obj, (dict, frozendict)):
        return {k: listify(v) for k, v in obj.items()}
    return obj


dict_keys = st.one_of(st.text(max_size=6), st.integers(min_value=-5, max_value=5))


def json_trees(max_leaves: int = 25):
    def containers(children):
        lists = st.lists(children, max_size=4)
        dicts = st.dictionaries(dict_keys, children, max_size=4)
        return st.one_of(lists, lists.map(tuple), dicts)

    return st.recursive(scalars, containers, max_leaves=max_leaves)


hashable_values = st.one_of(scalars, st.tuples(scalars, scalars))
frozendict_dicts = st.dictionaries(dict_keys, hashable_values, max_size=6)
unhashable_values = st.one_of(
    st.lists(scalars, max_size=3), st.dictionaries(dict_keys, scalars, max_size=3)
)
unhashable_frozendicts = st.dictionaries(
    dict_keys, unhashable_values, min_size=1, max_size=4
).map(frozendict)
