"""striqt.analysis.lib.register: measurement registration hooks, the validation and
tolerance walks over an analysis group, and the contract every registered measurement
follows (dtype, dims, coords, and attrs reach the DataArray).

The hook tests build a private `AnalysisRegistry`.
"""

from __future__ import annotations

import weakref
from fractions import Fraction
from typing import Optional

import msgspec
import numpy as np
import pytest
from analysis_strategies import (
    POWER_BINS,
    VALIDATOR_DOMAINS,
    RecordingIQ,
    capture_of,
    registered_tolerance,
)
from hypothesis import given, settings, strategies as st

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib import dataarrays, register
from striqt.analysis.specs.helpers import SpecValidationError

CAPTURE = sa.specs.Capture(duration=1e-4, sample_rate=1e6)


class ToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    nfft: int = 8


class OtherToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    scale: float = 1.0


class ToyGroup(sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True):
    toy: Optional[ToySpec] = None
    other_toy: Optional[OtherToySpec] = None


def build_registry(validate=None, *, spec_type=ToySpec, calls=None, tolerance=None):
    """return a registry with one measurement of `spec_type`, and the wrapped
    measurement.

    When `calls` is a list, each run of the measurement body appends
    ``('body', capture, kwargs)``.
    """
    registry = register.AnalysisRegistry()

    @registry.measurement(
        spec_type,
        dtype='float32',
        standard_name='Toy',
        long_name='Toy',
        dims=('toy',),
        validate=validate,
        tolerance=tolerance,
    )
    def toy(iq, capture, **kwargs):
        if calls is not None:
            calls.append(('body', capture, kwargs))
        return np.zeros((1, 2), dtype='float32')

    return registry, toy


# %% prefer_filtered= and prefer_aligned= on measurement registration


def build_preference_registry(**preferences: bool):
    """return a registry with one `ToySpec` measurement registered with `preferences`,
    the wrapped measurement, and the list of arrays passed to the body."""
    registry = register.AnalysisRegistry()
    received = []

    @registry.measurement(
        ToySpec,
        dtype='complex64',
        standard_name='Toy',
        long_name='Toy',
        dims=('toy',),
        **preferences,
    )
    def toy(x, capture, **kwargs):
        received.append(x)
        return x

    return registry, toy, received


@pytest.mark.parametrize(
    ('preferences', 'expected'),
    [
        ({}, (True, True)),
        ({'prefer_filtered': False, 'prefer_aligned': False}, (False, False)),
    ],
    ids=['default', 'explicit'],
)
def test_registration_records_the_preferences(preferences, expected):
    registry, _, _ = build_preference_registry(**preferences)
    info = registry[ToySpec]
    assert (info.prefer_filtered, info.prefer_aligned) == expected


def test_the_wrapper_hands_the_body_the_preprocessed_array():
    _, toy, received = build_preference_registry(
        prefer_filtered=False, prefer_aligned=True
    )
    x = np.arange(8, dtype='complex64').reshape(1, 8)
    iq = RecordingIQ(x=x, capture=CAPTURE, power_units='a.u.')

    toy(iq, CAPTURE, as_xarray=False)

    assert iq.preprocess.calls == [(False, True)]
    (body_input,) = received
    assert np.array_equal(body_input, x + 1)


def test_the_wrapper_passes_a_bare_array_through():
    _, toy, received = build_preference_registry(prefer_filtered=False)
    x = np.arange(8, dtype='complex64').reshape(1, 8)

    toy(x, CAPTURE, as_xarray=False)

    (body_input,) = received
    assert body_input is x


def test_signal_trigger_rejects_a_measurement_that_prefers_aligned_input():
    registry, toy, _ = build_preference_registry(prefer_aligned=True)

    def lag(capture, spec):
        return np.arange(2)

    with pytest.raises(ValueError, match='prefer_aligned=False'):
        registry.signal_trigger(ToySpec, lag_coord_func=lag)(toy)
    assert not registry.signal_trigger


def test_signal_trigger_accepts_a_measurement_that_prefers_unaligned_input():
    registry, toy, _ = build_preference_registry(prefer_aligned=False)

    def lag(capture, spec):
        return np.arange(2)

    registry.signal_trigger(ToySpec, lag_coord_func=lag)(toy)
    assert registry.signal_trigger[toy].meas_spec_type is ToySpec


# %% validate= on measurement registration


def test_validate_receives_the_projected_capture_and_the_decoded_spec():
    seen = []

    def validate(capture, spec):
        seen.append((capture, spec))

    _, toy = build_registry(validate)
    toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False, nfft=16)

    ((capture, spec),) = seen
    assert type(capture) is sa.specs.AnalysisCapture
    assert capture == sa.specs.helpers.to_analysis_capture(CAPTURE)
    assert spec == ToySpec(nfft=16)


def test_validate_runs_before_the_measurement_body():
    calls = []

    def validate(capture, spec):
        calls.append(('validate', capture, spec))

    _, toy = build_registry(validate, calls=calls)
    toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert [name for name, *_ in calls] == ['validate', 'body']


def test_a_validator_failure_skips_the_measurement_body():
    calls = []

    def validate(capture, spec):
        raise ValueError('no')

    _, toy = build_registry(validate, calls=calls)
    with pytest.raises(SpecValidationError):
        toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert calls == []


@pytest.mark.parametrize(
    'raised',
    [
        ValueError('nfft too large'),
        TypeError('nfft too large'),
        SpecValidationError('nfft too large', ('.nfft',)),
    ],
    ids=['value_error', 'type_error', 'spec_validation_error'],
)
def test_a_validator_failure_is_annotated_with_the_measurement_name(raised):
    def validate(capture, spec):
        raise raised

    _, toy = build_registry(validate)
    with pytest.raises(SpecValidationError) as excinfo:
        toy(np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False)

    assert excinfo.value.path[0] == '.toy'
    assert str(excinfo.value).startswith('$.toy')
    assert str(excinfo.value).endswith(': nfft too large')


def test_validate_defaults_to_none_and_the_measurement_still_runs():
    registry, toy = build_registry()
    assert registry[ToySpec].validate is None

    data, _ = toy(
        np.zeros((1, 100), dtype='complex64'), CAPTURE, as_xarray=False, nfft=16
    )
    assert data.shape == (1, 2)


# %% AnalysisRegistry.validate


def build_group_registry(*validators, tolerances=(None, None)):
    """return a registry of 2 measurements, `ToySpec` and `OtherToySpec`, with the
    validator and tolerance given for each."""
    registry = register.AnalysisRegistry()
    toy_validate, other_validate = validators
    toy_tolerance, other_tolerance = tolerances

    @registry.measurement(
        ToySpec,
        dtype='float32',
        standard_name='Toy',
        long_name='Toy',
        dims=('toy',),
        validate=toy_validate,
        tolerance=toy_tolerance,
    )
    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    @registry.measurement(
        OtherToySpec,
        dtype='float32',
        standard_name='Other toy',
        long_name='Other toy',
        dims=('other',),
        validate=other_validate,
        tolerance=other_tolerance,
    )
    def other_toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    return registry


@pytest.mark.parametrize(
    ('toy_has_validator', 'group'),
    [
        (True, ToyGroup(other_toy=OtherToySpec())),
        (False, ToyGroup(toy=ToySpec(), other_toy=OtherToySpec())),
    ],
    ids=['toy_not_in_group', 'toy_without_validator'],
)
def test_registry_validate_runs_the_validators_of_the_group_on_the_projected_capture(
    toy_has_validator, group
):
    seen = []

    def recorder(name):
        return lambda capture, spec: seen.append((name, capture))

    registry = build_group_registry(
        recorder('toy') if toy_has_validator else None, recorder('other_toy')
    )

    registry.validate(CAPTURE, group)
    assert seen == [('other_toy', sa.specs.helpers.to_analysis_capture(CAPTURE))]


def test_registry_validate_annotates_the_analysis_field_path():
    def validate(capture, spec):
        raise SpecValidationError('nfft must be even', ('.nfft',))

    registry = build_group_registry(validate, None)
    group = ToyGroup(toy=ToySpec())

    with pytest.raises(SpecValidationError) as excinfo:
        registry.validate(CAPTURE, group)

    assert excinfo.value.path == ('.analysis', '.toy', '.nfft')
    assert str(excinfo.value) == '$.analysis.toy.nfft: nfft must be even'


def test_validator_errors_are_msgspec_validation_errors():
    """the sweep decode path reports validator errors as decode failures, so a
    `SpecValidationError` is a `msgspec.ValidationError`."""
    registry = build_group_registry(
        lambda capture, spec: (_ for _ in ()).throw(ValueError('nope')), None
    )
    group = ToyGroup(toy=ToySpec())

    with pytest.raises(msgspec.ValidationError):
        registry.validate(CAPTURE, group)


# %% tolerance= on measurement registration

TOY_TOLERANCE = sa.specs.Tolerance(
    units='dB', rtol=1e-6, on_peak=sa.specs.ErrorBound(rms=1e-3, peak=1e-2)
)


def recording_tolerance(seen):
    def tolerance(capture, spec, **kwargs):
        seen.append((capture, spec, kwargs))
        return TOY_TOLERANCE

    return tolerance


@pytest.mark.parametrize(
    ('declared', 'group', 'expected'),
    [
        ((False, False), ToyGroup(toy=ToySpec(), other_toy=OtherToySpec()), {}),
        (
            (False, True),
            ToyGroup(toy=ToySpec(), other_toy=OtherToySpec()),
            {'other_toy': TOY_TOLERANCE},
        ),
        ((True, True), ToyGroup(toy=ToySpec()), {'toy': TOY_TOLERANCE}),
    ],
    ids=['none_declared', 'one_declared', 'other_toy_not_in_group'],
)
def test_registry_tolerances_maps_the_declared_measurements_of_the_group(
    declared, group, expected
):
    tolerances = tuple(recording_tolerance([]) if d else None for d in declared)
    registry = build_group_registry(None, None, tolerances=tolerances)

    assert registry.tolerances(CAPTURE, group) == expected


def test_registry_tolerances_pass_the_projected_capture_spec_and_keywords():
    seen = []
    registry = build_group_registry(
        None, None, tolerances=(recording_tolerance(seen), None)
    )
    group = ToyGroup(toy=ToySpec(nfft=4))

    registry.tolerances(CAPTURE, group)
    registry.tolerances(CAPTURE, group, array_backend='cupy', input_error=1e-4)

    projected = sa.specs.helpers.to_analysis_capture(CAPTURE)
    assert [(capture, spec) for capture, spec, _ in seen] == [
        (projected, ToySpec(nfft=4))
    ] * 2
    assert [kws for *_, kws in seen] == [
        {'array_backend': 'numpy', 'input_error': 0.0},
        {'array_backend': 'cupy', 'input_error': 1e-4},
    ]


# %% a validator that passes means the measurement runs

VALIDATED = [info for info in sa.registry.values() if info.validate is not None]
VALIDATED_IDS = [info.name for info in VALIDATED]
# the SSS correlators share the PSS validator and search all 1008 cell ids per
# accepted example, so the PSS measurements alone draw from the shared domain
OTHERS = [info for info in VALIDATED if 'sss' not in info.name]


def test_every_validated_measurement_has_a_strategy():
    """every measurement with a validator has a domain in `VALIDATOR_DOMAINS`, and no
    domain names a measurement without a validator."""
    assert set(VALIDATOR_DOMAINS) == set(VALIDATED_IDS)


def check_accepted_pair_runs(info, data):
    """draw a (capture, spec) pair for `info` and fail the test when the validator
    accepts the pair but the measurement raises on white noise.

    `Sweep.__post_init__` runs the validators on every capture before acquisition, so
    an accepted pair must not raise on shape or scalar arithmetic once IQ arrives. A
    rejected pair returns without a check.
    """
    capture, spec = data.draw(VALIDATOR_DOMAINS[info.name], label=info.name)
    group = sa.specs.BuiltinAnalyses(**{info.name: spec})
    try:
        sa.registry.validate(capture, group)
    except SpecValidationError:
        return

    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    iq = sa.testing.circular_awgn(
        capture.duration, capture.sample_rate, power=1.0, ports=ports
    )
    try:
        info.func(iq, capture, as_xarray=False, **spec.to_dict())
    except (ValueError, IndexError, TypeError) as ex:
        pytest.fail(f'{info.name} accepted {capture!r} with {spec!r} but raised {ex!r}')


@pytest.mark.parametrize('info', OTHERS, ids=[i.name for i in OTHERS])
@given(data=st.data())
# the domains are finite `sampled_from` draws that 20 examples per measurement cover;
# 50 cost 0.9 s over the 12 measurements
@settings(max_examples=20)
def test_a_validator_that_passes_means_the_measurement_runs(info, data):
    check_accepted_pair_runs(info, data)


FS = 1.024e6
NFFT = 8
SAMPLES = 64
SPG = {'window': 'boxcar', 'frequency_resolution': FS / NFFT}

# (capture, spec, the spec or capture field the message must name). Every registered
# validator raises a bare ValueError, so `registry.validate` adds one step,
# `validation_path` wrapping the error at the measurement key. One row names a
# capture field and one a spec field. The per-measurement rejection tables are in the
# test_analysis_measurements_* modules.
REJECTED = {
    'spectrogram_shorter_than_nfft': (
        capture_of(NFFT - 1, FS),
        sa.specs.Spectrogram(**SPG),
        'duration',
    ),
    'psd_unknown_statistic': (
        capture_of(SAMPLES, FS),
        sa.specs.PowerSpectralDensity(**SPG, time_statistic=('mean', 'bogus')),
        'time_statistic',
    ),
}


@pytest.mark.parametrize('case', list(REJECTED), ids=list(REJECTED))
def test_validator_rejects_at_the_analysis_path_naming_the_field(case):
    """a pair the registered validator rejects is reported at the measurement key
    under `$.analysis`, and the message names the spec or capture field at fault."""
    capture, spec, field = REJECTED[case]
    name = sa.registry[type(spec)].name
    group = sa.specs.BuiltinAnalyses(**{name: spec})

    with pytest.raises(SpecValidationError) as excinfo:
        sa.registry.validate(capture, group)

    assert excinfo.value.path[:2] == ('.analysis', f'.{name}')
    assert str(excinfo.value).startswith(f'$.analysis.{name}')
    assert field in excinfo.value.message


# %% the contract shared by every measurement

CONTRACT_FS = 210e3
CONTRACT_SCS = 15e3
CONTRACT_DETECTOR_PERIOD = Fraction(1, 10000)
CONTRACT_CAPTURE = sa.specs.Capture(
    duration=4e-3, sample_rate=CONTRACT_FS, analysis_bandwidth=150e3
)

# 4 ms at 210 kHz is 1 discovery period of 56 symbols at 15 kHz subcarriers and a
# whole number of detector and cyclic periods. The first 28 symbols are the SSB burst
# set.
CONTRACT_SPECS = (
    sa.specs.Spectrogram(window='hamming', frequency_resolution=7.5e3),
    sa.specs.PowerSpectralDensity(window='hamming', frequency_resolution=7.5e3),
    sa.specs.Cellular5GNRSSBSpectrogram(
        subcarrier_spacing=CONTRACT_SCS,
        sample_rate=CONTRACT_FS / 2,
        discovery_periodicity=CONTRACT_CAPTURE.duration,
    ),
    sa.specs.ChannelPowerTimeSeries(detector_period=CONTRACT_DETECTOR_PERIOD),
    sa.specs.CyclicChannelPower(
        cyclic_period=1e-3, detector_period=CONTRACT_DETECTOR_PERIOD
    ),
    sa.specs.IQWaveform(),
    sa.specs.ChannelPowerHistogram(
        detector_period=CONTRACT_DETECTOR_PERIOD, **POWER_BINS
    ),
    sa.specs.SpectrogramHistogram(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.SpectrogramHistogramRatio(
        window='hamming', frequency_resolution=7.5e3, **POWER_BINS
    ),
    sa.specs.CellularResourcePowerHistogram(
        window='hamming', subcarrier_spacing=CONTRACT_SCS, **POWER_BINS
    ),
)

SPEC_IDS = [type(spec).__name__ for spec in CONTRACT_SPECS]


def measure(spec, as_xarray):
    """return the registration and the result of the measurement registered for
    `type(spec)` on a unit tone over `CONTRACT_CAPTURE`, acquired in mW."""
    info = sa.registry[type(spec)]
    ports = 2 if isinstance(spec, sa.specs.SpectrogramHistogramRatio) else 1
    x = sa.testing.tone(
        CONTRACT_CAPTURE.duration, CONTRACT_CAPTURE.sample_rate, ports=ports
    )
    iq = sa.AcquiredIQ(x, CONTRACT_CAPTURE, power_units='mW')
    return info, info.func(iq, CONTRACT_CAPTURE, as_xarray=as_xarray, **spec.to_dict())


def registered_dims(info):
    dims = ['port']
    for factory in info.coord_factories:
        for dim in sa.registry.coordinates[factory].dims:
            if dim not in dims:
                dims.append(dim)
    return tuple(dims)


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_dtype_and_dims(spec):
    info, da = measure(spec, as_xarray=True)

    assert da.dtype == info.dtype
    if info.dims is None:
        assert da.dims == registered_dims(info)
    else:
        assert da.dims == ('port',) + info.dims
    assert set(da.coords) == set(da.dims) - {'port'}


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_attrs(spec):
    """the registered attrs and every spec field reach the DataArray attrs. A saved
    zarr store identifies a measurement through those attrs."""
    info, da = measure(spec, as_xarray=True)

    for name, value in info.attrs.items():
        if name == 'units':
            value = sw.units.resolve_power_units(value, 'mW')
        assert da.attrs[name] == value

    # the wrapper decodes the keywords into a spec before calling, so the attrs are
    # the validated field values rather than the ones passed in
    for name, value in spec.validate().to_dict().items():
        assert da.attrs[name] == value


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_registered_power_units_are_form_tokens(spec):
    """a registered or returned `units` that resolves to a power family is declared
    as a form token, never as a literal family."""
    info, (_, body_attrs) = measure(spec, as_xarray=False)
    for attrs, what in [(info.attrs, info.name), (body_attrs, f'{info.name} body')]:
        units = attrs.get('units')
        if not isinstance(units, str):
            continue
        resolved = sw.units.resolve_power_units(units, 'mW')
        if sw.units.power_family(resolved) is not None:
            assert sw.units.has_power_form(units), what
    for factory in info.coord_factories:
        coord_info = sa.registry.coordinates[factory]
        units = coord_info.attrs.get('units')
        if isinstance(units, str) and sw.units.power_family(units) is not None:
            raise AssertionError(f'{coord_info.name} declares the literal {units!r}')


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_absolute_power_outputs_name_their_noise_bandwidth(spec):
    """every output and power-bin coordinate in a power family (`power_family` is not
    None) has a `noise_bandwidth` (Hz), `units` without a bandwidth, and a
    `standard_name` without a parenthesized suffix. The outputs with the
    per-capture `analysis_bandwidth` as noise bandwidth set no `noise_bandwidth`.
    Ratios in dB set no `noise_bandwidth`."""
    info, da = measure(spec, as_xarray=True)

    def check(attrs, what):
        units = attrs.get('units')
        if str(what) in sa.lib.io._CAPTURE_BANDWIDTH_NAMES:
            assert 'noise_bandwidth' not in attrs, what
        elif isinstance(units, str) and sw.units.power_family(units) is not None:
            assert isinstance(attrs.get('noise_bandwidth'), float), what
            assert attrs['noise_bandwidth'] > 0, what
            assert 'dBm/' not in units, what
            assert '(' not in attrs.get('standard_name', ''), what
        elif units == 'dB':
            assert 'noise_bandwidth' not in attrs, what

    check(da.attrs, info.name)
    for name in da.coords:
        if str(name).endswith('_power_bin'):
            check(da[name].attrs, name)
            assert ('noise_bandwidth' in da[name].attrs) == (
                'noise_bandwidth' in da.attrs
            ), name


@pytest.mark.parametrize('spec', CONTRACT_SPECS, ids=SPEC_IDS)
def test_measurement_geometry_matches_the_measured_dataarray(spec):
    """the dims, shape, coordinate values, and coordinate attrs that a sink computes
    from ``(capture, spec)`` before any acquisition equal those of the measured
    DataArray."""
    info, da = measure(spec, as_xarray=True)
    spec = spec.validate()

    assert dataarrays.measurement_dims(info) == da.dims[1:]
    assert dataarrays.measurement_shape(info, CONTRACT_CAPTURE, spec) == da.shape[1:]

    coords = dataarrays.measurement_coords(info, CONTRACT_CAPTURE, spec, 'mW')
    assert set(coords) == set(da.coords)
    for name, (values, attrs) in coords.items():
        assert values.dtype == da[name].dtype, name
        assert np.array_equal(
            values, da[name].values, equal_nan=values.dtype.kind in 'fc'
        ), name
        assert attrs == da[name].attrs, name


def test_measurement_shape_rejects_a_dim_without_a_coordinate_factory():
    info = register.AnalysisInfo(
        name='toy',
        func=lambda iq, capture, **kwargs: None,
        coord_factories=(),
        cache=None,
        dtype='float32',
        attrs={},
        depends=(),
        store_compressed=True,
        prefer_filtered=True,
        prefer_aligned=True,
        dims=('toy',),
    )

    with pytest.raises(NotImplementedError, match="'toy'"):
        dataarrays.measurement_shape(info, CONTRACT_CAPTURE, ToySpec())


# the only measurement without a validator: iq_waveform clamps the start/stop bounds
# into the capture rather than rejecting an out-of-range bound
NO_VALIDATOR = {'iq_waveform'}


def test_every_measurement_has_a_validator():
    """a measurement without a validator skips pre-sweep validation, so every
    registered measurement has `validate=` or is listed in `NO_VALIDATOR`."""
    missing = {
        info.name for info in sa.registry.values() if info.validate is None
    } - NO_VALIDATOR

    assert missing == set()
    assert NO_VALIDATOR <= {info.name for info in sa.registry.values()}


CONTRACT_TOLERANCE_SPECS = [
    s for s in CONTRACT_SPECS if sa.registry[type(s)].tolerance is not None
]
CONTRACT_TOLERANCE_IDS = [type(spec).__name__ for spec in CONTRACT_TOLERANCE_SPECS]


@pytest.mark.parametrize('spec', CONTRACT_TOLERANCE_SPECS, ids=CONTRACT_TOLERANCE_IDS)
def test_registered_tolerance_is_a_dB_budget_that_grows_with_input_error(spec):
    """the registered budget bounds a level in dB with a peak at or above the rms.
    An amplitude error in the IQ widens both."""
    exact = registered_tolerance(CONTRACT_CAPTURE, spec, input_error=0.0)
    assert isinstance(exact, sa.specs.Tolerance)
    assert exact.units == 'dB'
    # iq_waveform copies the input, so the exact-input bound of iq_waveform is 0
    assert exact.on_peak.peak >= exact.on_peak.rms >= 0

    perturbed = registered_tolerance(CONTRACT_CAPTURE, spec, input_error=1e-4)
    assert perturbed.on_peak.rms > exact.on_peak.rms
    assert perturbed.on_peak.peak > exact.on_peak.peak


# %% check_group_fields and check_group_registered


class SubsetGroup(
    sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    toy: Optional[ToySpec] = None


class RenamedGroup(
    sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    renamed: Optional[ToySpec] = None


class KeptDefaultsGroup(sa.specs.AnalysisGroup, frozen=True, kw_only=True):
    toy: Optional[ToySpec] = None
    other_toy: Optional[OtherToySpec] = None


class NotOptionalGroup(
    sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    toy: ToySpec = ToySpec()
    other_toy: Optional[OtherToySpec] = None


TOY_FIELDS = {'toy': ToySpec, 'other_toy': OtherToySpec}


def check_fields(group, expected=TOY_FIELDS):
    return register.check_group_fields(group, expected)


def check_registered(group, registry=None):
    if registry is None:
        registry = build_group_registry(None, None)
    return register.check_group_registered(group, registry)


# (check, the TypeError message a rejection must match, or None when accepted). Only a
# group with ``omit_defaults=True`` omits the disabled measurements from `to_dict` and
# the saved attrs, so both checks reject a group without it.
GROUP_CHECKS = {
    'fields_match': (lambda: check_fields(ToyGroup), None),
    'fields_missing': (
        lambda: check_fields(ToyGroup, {**TOY_FIELDS, 'third': ToySpec}),
        r"missing fields \['third'\]",
    ),
    'fields_unregistered': (
        lambda: check_fields(ToyGroup, {'toy': ToySpec}),
        r"unregistered fields \['other_toy'\]",
    ),
    'fields_wrong_type': (
        lambda: check_fields(
            ToyGroup, {'toy': OtherToySpec, 'other_toy': OtherToySpec}
        ),
        r"wrong spec type \['toy'\]",
    ),
    'fields_not_optional': (lambda: check_fields(NotOptionalGroup), 'Optional'),
    'fields_without_omit_defaults': (
        lambda: check_fields(KeptDefaultsGroup),
        'omit_defaults',
    ),
    'registered_match': (lambda: check_registered(ToyGroup), None),
    'registered_subset': (lambda: check_registered(SubsetGroup), None),
    'registered_unregistered_type': (
        lambda: check_registered(ToyGroup, build_registry()[0]),
        'other_toy: OtherToySpec is not a registered',
    ),
    'registered_renamed_field': (
        lambda: check_registered(RenamedGroup),
        "renamed must be named 'toy'",
    ),
    'registered_not_optional': (lambda: check_registered(NotOptionalGroup), 'Optional'),
    'registered_without_omit_defaults': (
        lambda: check_registered(KeptDefaultsGroup),
        'omit_defaults',
    ),
}


@pytest.mark.parametrize('case', list(GROUP_CHECKS), ids=list(GROUP_CHECKS))
def test_group_checks_accept_matching_groups_and_name_the_rejected_field(case):
    check, match = GROUP_CHECKS[case]
    if match is None:
        assert check() is None
    else:
        with pytest.raises(TypeError, match=match):
            check()


def test_registry_and_any_analysis_agree():
    """the import-time guard of `striqt.analysis.measurements`: every registered
    measurement is a `BuiltinAnalyses` field of the same name, and no other field
    exists."""
    expected = {info.name: spec_type for spec_type, info in sa.registry.items()}
    register.check_group_fields(sa.specs.BuiltinAnalyses, expected)
    assert set(sa.specs.BuiltinAnalyses.__struct_fields__) == set(expected)


def test_trigger_registry_and_any_signal_trigger_agree():
    expected = {
        info.name: info.meas_spec_type
        for key, info in sa.registry.signal_trigger.items()
        if isinstance(key, str)
    }
    register.check_group_fields(sa.specs.BuiltinTriggers, expected)
    triggers = set(sa.specs.BuiltinTriggers.__struct_fields__)
    assert triggers <= set(sa.specs.BuiltinAnalyses.__struct_fields__)


# %% KwArgCache


def build_counting_cache(maxsize=4):
    """return a `KwArgCache` keyed on `nfft`, the wrapped function, and the list of
    ``(id(iq), nfft)`` calls the function records."""
    cache = register.KwArgCache(['nfft'], maxsize=maxsize)
    calls = []

    @cache.apply
    def helper(iq, capture, *, nfft):
        calls.append((id(iq), nfft))
        return np.zeros(nfft)

    return cache, helper, calls


def test_kwargcache_hits_on_the_same_array_and_keywords():
    cache, helper, calls = build_counting_cache()
    x = np.ones((1, 4))

    with cache:
        first = helper(x, CAPTURE, nfft=8)
        second = helper(x, CAPTURE, nfft=8)

    assert second is first
    assert len(calls) == 1


def test_kwargcache_misses_on_a_different_array_with_the_same_keywords():
    cache, helper, calls = build_counting_cache()
    x = np.ones((1, 4))
    y = x.copy()

    with cache:
        helper(x, CAPTURE, nfft=8)
        helper(y, CAPTURE, nfft=8)

    assert [iq_id == id(y) for iq_id, _ in calls] == [False, True]


def test_kwargcache_holds_the_array_until_exit():
    cache, helper, _ = build_counting_cache()
    x = np.ones((1, 4))
    ref = weakref.ref(x)

    with cache:
        helper(x, CAPTURE, nfft=8)
        del x
        assert ref() is not None

    assert ref() is None


def test_kwargcache_holds_several_keyword_values():
    """results for 2 keyword values alternate without recomputation"""
    cache, helper, calls = build_counting_cache()
    x = np.ones((1, 4))

    with cache:
        first = helper(x, CAPTURE, nfft=8)
        helper(x, CAPTURE, nfft=16)
        assert helper(x, CAPTURE, nfft=8) is first

    assert [nfft for _, nfft in calls] == [8, 16]


def test_kwargcache_maxsize_evicts_the_oldest_entry():
    cache, helper, calls = build_counting_cache(maxsize=1)
    x = np.ones((1, 4))

    with cache:
        helper(x, CAPTURE, nfft=8)
        helper(x, CAPTURE, nfft=16)
        helper(x, CAPTURE, nfft=8)

    assert [nfft for _, nfft in calls] == [8, 16, 8]


# %% registration keywords


@pytest.mark.parametrize('given', ['standard_name', 'long_name'])
def test_registration_requires_both_names(given):
    registry = register.AnalysisRegistry()
    with pytest.raises(TypeError):
        registry.measurement(ToySpec, dtype='float32', dims=('toy',), **{given: 'Toy'})  # ty: ignore[missing-argument]


def test_registered_attrs_are_the_names_and_units():
    """the attrs of a registration are `standard_name`, `long_name`, and `units` when
    given, and a `units` of None sets no key."""
    registry = register.AnalysisRegistry()

    @registry.measurement(
        ToySpec, dtype='float32', standard_name='Toy', long_name='Toy', dims=('toy',)
    )
    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    @registry.measurement(
        OtherToySpec,
        dtype='float32',
        standard_name='Other toy',
        long_name='Other toy',
        units=sa.units.DB_POWER,
        dims=('other',),
    )
    def other_toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    assert registry[ToySpec].attrs == {'standard_name': 'Toy', 'long_name': 'Toy'}
    assert registry[OtherToySpec].attrs == {
        'standard_name': 'Other toy',
        'long_name': 'Other toy',
        'units': sa.units.DB_POWER,
    }
    assert registry[ToySpec].name == 'toy'
    assert registry[OtherToySpec].name == 'other_toy'


def test_coordinate_registration_names_itself_and_keeps_its_units():
    coords = register.CoordRegistry()

    @coords(dtype='float64', standard_name='Toy lag', units='s')
    def toy_lag(capture, spec):
        return np.arange(2.0)

    info = coords[toy_lag]
    assert info.name == 'toy_lag'
    assert info.dims == ('toy_lag',)
    assert info.attrs == {'standard_name': 'Toy lag', 'units': 's'}
    with pytest.raises(KeyError):
        coords(dtype='float64', standard_name='Again')(toy_lag)


# %% generated docstrings

MEASUREMENT_DOC_TEMPLATE = """summary.

    Output:
        - Array dtype: {dtype}
        - Dimensions: {dims}
        - Coordinates: {coords}
        - Attributes: {attrs}
          * `noise_bandwidth`: a hand-written item after the generated ones

    A literal {dB} form field survives.

    Args:
    {args}
    """


def test_measurement_docstring_fills_dims_and_coords_in_place():
    registry = register.AnalysisRegistry()

    @registry.coordinates(dtype='float64', standard_name='Toy lag', units='s')
    def toy_lag(capture, spec):
        """time from the start,
        in steps of `period`.

        Returns:
            attrs that the registry does not render
        """
        return np.arange(2.0)

    @registry.coordinates(
        dtype='float32', standard_name='Bare axis', units=sa.units.DB_POWER
    )
    def bare(capture, spec):
        return np.arange(2.0, dtype='float32')

    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2, 2), dtype='float32')

    toy.__doc__ = MEASUREMENT_DOC_TEMPLATE
    wrapped = registry.measurement(
        ToySpec,
        dtype='float32',
        standard_name='Toy',
        long_name='Toy',
        units=sa.units.DB_POWER,
        coord_factories=[toy_lag, bare],
    )(toy)
    doc = wrapped.__doc__

    for placeholder in ('{dtype}', '{dims}', '{coords}', '{attrs}', '{args}'):
        assert placeholder not in doc
    assert '- Array dtype: float32' in doc
    assert '- Dimensions: `port`, `toy_lag`, `bare`' in doc
    # the first paragraph of the factory docstring, whitespace collapsed
    assert '* `toy_lag` (float64, s): time from the start, in steps of `period`' in doc
    # an undocumented factory falls back to its standard_name; a power form is
    # rendered as the calibrated unit
    assert '* `bare` (float32, dBm when calibrated): Bare axis' in doc
    assert 'attrs that the registry does not render' not in doc
    units_item = '* `units`: power units based on acquisition (dBm when calibrated)'
    assert units_item in doc
    # the generated items precede the hand-written ones
    assert doc.index(units_item) < doc.index('* `noise_bandwidth`:')
    assert 'A literal {dB} form field survives.' in doc
    assert 'Args:' in doc
    assert 'iq (Array or AcquiredIQ)' in doc
    assert 'nfft (int)' in doc


def test_measurement_docstring_with_explicit_dims_and_no_factories():
    registry = register.AnalysisRegistry()

    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    toy.__doc__ = (
        'summary.\n\n    - Dimensions: {dims}\n    - Coordinates: {coords}\n'
        '    - Attributes: {attrs}\n'
    )
    wrapped = registry.measurement(
        ToySpec, dtype='float32', standard_name='Toy', long_name='Toy', dims=('toy',)
    )(toy)

    assert '- Dimensions: `port`, `toy`' in wrapped.__doc__
    assert '- Coordinates: none' in wrapped.__doc__
    assert '- Attributes:' in wrapped.__doc__
    assert '{attrs}' not in wrapped.__doc__


def test_measurement_without_a_docstring_gets_the_args_section():
    registry = register.AnalysisRegistry()

    @registry.measurement(
        ToySpec, dtype='float32', standard_name='Toy', long_name='Toy', dims=('toy',)
    )
    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    assert toy.__doc__.startswith('Args:')
    assert 'iq (Array or AcquiredIQ)' in toy.__doc__
    assert 'nfft (int)' in toy.__doc__
