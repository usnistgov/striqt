"""SpecBase freezing and Capture validation in __post_init__"""

from __future__ import annotations

import math

import msgspec
import pytest
from conftest import raises_on_both_paths
from hypothesis import HealthCheck, given, settings
from spec_strategies import fractional_sample_captures, integer_sample_captures

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import frozendict, inspect_freeze_depths

Capture = sa.specs.Capture
CCA = sa.specs.CellularCyclicAutocorrelator
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)
PERIOD_MSG = 'is not an integer multiple of sample period'


class TestSpecBaseFreezing:
    def test_fixed_tuple_field_list_is_frozen_on_direct_construction(self):
        spec = sa.specs.CellularResourcePowerHistogram(
            window='hann',
            subcarrier_spacing=15e3,
            power_low=-100,
            power_high=0,
            power_resolution=1,
            guard_bandwidths=[1, 2],
        )
        assert spec.guard_bandwidths == (1, 2)
        assert isinstance(spec.guard_bandwidths, tuple)
        hash(spec)

    def test_dict_field_is_frozen_to_frozendict(self, construct):
        spec = construct(ss.specs.DataOptions, sweep_index=0, select={'a': [1]})
        assert isinstance(spec.select, frozendict)
        assert spec.select == {'a': (1,)}
        hash(spec)

    def test_inspect_freeze_depths_skips_var_tuple(self):
        # subcarrier_spacings is tuple[float, ...], which the depth inspection
        # does not recognize as a container
        assert inspect_freeze_depths(CCA) == {'frame_range': 1, 'symbol_range': 1}

    @pytest.mark.xfail(
        strict=True,
        reason=(
            '_inspect_container_depth handles TupleType but not VarTupleType, so '
            'tuple[float, ...] fields are skipped by SpecBase.__post_init__'
        ),
    )
    def test_var_tuple_list_is_frozen_on_direct_construction(self):
        spec = CCA(subcarrier_spacings=[15e3, 30e3])
        assert spec.subcarrier_spacings == (15e3, 30e3)
        hash(spec)

    def test_var_tuple_list_is_frozen_by_from_dict(self):
        spec = CCA.from_dict({'subcarrier_spacings': [15e3, 30e3]})
        assert spec.subcarrier_spacings == (15e3, 30e3)
        hash(spec)

    def test_unhashable_direct_struct_cannot_validate_or_replace(self):
        spec = CCA(subcarrier_spacings=[15e3])
        with pytest.raises(TypeError, match='unhashable'):
            spec.validate()
        with pytest.raises(TypeError, match='unhashable'):
            spec.replace(frame_range=(0, 2))
        # the msgspec replace itself is fine; only the cached validate step hashes
        msgspec.structs.replace(spec, frame_range=(0, 2))


class TestCapture:
    @given(kws=integer_sample_captures())
    @PROPERTY
    def test_integer_sample_count_is_accepted(self, construct, kws):
        capture = construct(Capture, **kws)
        assert capture.duration == kws['duration']
        assert capture.sample_rate == kws['sample_rate']

    @given(kws=fractional_sample_captures())
    @PROPERTY
    def test_fractional_sample_count_is_rejected(self, kws):
        raises_on_both_paths(Capture, ValueError, PERIOD_MSG, **kws)

    def test_replace_revalidates_duration(self):
        capture = Capture(duration=1e-3, sample_rate=1e6)
        with pytest.raises(ValueError, match=PERIOD_MSG):
            capture.replace(duration=1.5e-6)

    def test_replace_without_attrs_is_identity(self):
        capture = Capture(duration=1e-3, sample_rate=1e6)
        assert capture.replace() is capture

    @pytest.mark.parametrize('excess_samples, ok', [(1e-7, True), (0.5, False)])
    def test_tolerance_is_one_microsample(self, excess_samples, ok):
        duration = (1000 + excess_samples) / 1e6
        if ok:
            Capture(duration=duration, sample_rate=1e6)
        else:
            with pytest.raises(ValueError, match=PERIOD_MSG):
                Capture(duration=duration, sample_rate=1e6)

    def test_nan_sample_rate_gives_sample_period_message(self):
        raises_on_both_paths(
            Capture, ValueError, PERIOD_MSG, duration=1e-3, sample_rate=math.nan
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'math.remainder(inf, 1) raises "math domain error" inside util.isroundmod '
            'before Capture can report its own message'
        ),
    )
    def test_infinite_sample_rate_reports_sample_period_message(self):
        with pytest.raises(ValueError, match=PERIOD_MSG):
            Capture(duration=1e-3, sample_rate=math.inf)

    @pytest.mark.parametrize(
        'kws',
        [{'duration': 0, 'sample_rate': 1e6}, {'duration': 1e-3, 'sample_rate': -1e6}],
    )
    def test_zero_duration_and_negative_sample_rate_pass(self, construct, kws):
        # the analysis Capture has no positivity constraints; those live on the
        # sensor-side backend rate types
        construct(Capture, **kws)

    def test_json_decode_runs_post_init(self):
        with pytest.raises(msgspec.ValidationError, match=PERIOD_MSG):
            msgspec.json.decode(
                b'{"duration": 1.5e-6, "sample_rate": 1e6}', type=Capture
            )
