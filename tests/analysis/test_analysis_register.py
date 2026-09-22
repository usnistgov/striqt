"""striqt.analysis.lib.register: the `validate=` hook on measurement registration
and the `AnalysisRegistry.validate` walk over an analysis group.

Everything here builds its own `AnalysisRegistry`; mutating the live
`register.registry` would leak a measurement into every other test module. The
measurements are called with `as_xarray=False` because the xarray path resolves
coordinate factories through the *global* registry.
"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest

import striqt.analysis as sa
from striqt.analysis.lib import register
from striqt.analysis.specs.helpers import SpecValidationError

CAPTURE = sa.specs.Capture(duration=1e-4, sample_rate=1e6)


class ToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    nfft: int = 8


class OtherToySpec(sa.specs.Analysis, frozen=True, kw_only=True):
    scale: float = 1.0


def build_registry(validate=None, *, spec_type=ToySpec, calls=None):
    """a registry holding one measurement that records its (capture, spec) calls"""
    registry = register.AnalysisRegistry()

    @registry.measurement(spec_type, dtype='float32', dims=('toy',), validate=validate)
    def toy(iq, capture, **kwargs):
        if calls is not None:
            calls.append(('body', capture, kwargs))
        return np.zeros((1, 2), dtype='float32')

    return registry, toy


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


def build_group_registry(*validators):
    """a registry of two measurements, each with the given validator"""
    registry = register.AnalysisRegistry()
    toy_validate, other_validate = validators

    @registry.measurement(
        ToySpec, dtype='float32', dims=('toy',), validate=toy_validate
    )
    def toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    @registry.measurement(
        OtherToySpec, dtype='float32', dims=('other',), validate=other_validate
    )
    def other_toy(iq, capture, **kwargs):
        return np.zeros((1, 2), dtype='float32')

    return registry


def test_registry_validate_calls_only_the_requested_measurements():
    seen = []
    registry = build_group_registry(
        lambda capture, spec: seen.append('toy'),
        lambda capture, spec: seen.append('other_toy'),
    )
    group = registry.tospec()(other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_skips_measurements_without_a_validator():
    seen = []
    registry = build_group_registry(
        None, lambda capture, spec: seen.append('other_toy')
    )
    group = registry.tospec()(toy=ToySpec(), other_toy=OtherToySpec())

    registry.validate(CAPTURE, group)
    assert seen == ['other_toy']


def test_registry_validate_annotates_the_analysis_field_path():
    def validate(capture, spec):
        raise SpecValidationError('nfft must be even', ('.nfft',))

    registry = build_group_registry(validate, None)
    group = registry.tospec()(toy=ToySpec())

    with pytest.raises(SpecValidationError) as excinfo:
        registry.validate(CAPTURE, group)

    assert excinfo.value.path == ('.analysis', '.toy', '.nfft')
    assert str(excinfo.value) == '$.analysis.toy.nfft: nfft must be even'


def test_registry_validate_passes_the_projected_capture():
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(capture), None)
    group = registry.tospec()(toy=ToySpec())

    registry.validate(CAPTURE, group)
    assert seen == [sa.specs.helpers.to_analysis_capture(CAPTURE)]


def test_registry_validate_is_cached_on_the_projected_capture():
    """the sensor walker relies on this to keep a long sweep off the critical path"""
    seen = []
    registry = build_group_registry(lambda capture, spec: seen.append(spec), None)
    group = registry.tospec()(toy=ToySpec())

    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE, group)
    registry.validate(CAPTURE.replace(analysis_bandwidth=5e5), group)

    assert len(seen) == 2


def test_registry_is_hashable():
    registry = build_group_registry(None, None)
    assert hash(registry) == hash(registry)
    assert isinstance(hash(registry), int)


def test_validator_errors_are_msgspec_validation_errors():
    """the sweep decode path reports these, so they must survive msgspec's own
    error handling and be catchable as a decode failure"""
    registry = build_group_registry(
        lambda capture, spec: (_ for _ in ()).throw(ValueError('nope')), None
    )
    group = registry.tospec()(toy=ToySpec())

    with pytest.raises(msgspec.ValidationError):
        registry.validate(CAPTURE, group)
