"""striqt.analysis.specs.structs: SpecBase freezing and __post_init__ validation"""

from __future__ import annotations

import math

import msgspec
import pytest
from conftest import raises_on_both_paths
from hypothesis import HealthCheck, given, settings
from spec_strategies import (
    descending_ranges,
    fractional_delay_kwargs,
    fractional_sample_captures,
    integer_delay_kwargs,
    integer_sample_captures,
    valid_frame_ranges,
    valid_symbol_ranges,
)

import striqt.analysis as sa

S = sa.specs
Capture = S.Capture
CCA = S.CellularCyclicAutocorrelator
CORRELATORS = (
    S.Cellular5GNRPSSCorrelator,
    S.Cellular5GNRSSSCorrelator,
    S.Cellular5GNPSSSync,
    S.Cellular5GNSSSSync,
)
SCS = 30e3
PERIOD_MSG = 'is not an integer multiple of sample period'
DELAY_MSG = r'delay \* sample_rate must be an integer'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

by_correlator = pytest.mark.parametrize('cls', CORRELATORS, ids=lambda c: c.__name__)


# %% SpecBase


class TestSpecBaseFreezing:
    def test_fixed_tuple_field_list_is_frozen_on_direct_construction(self):
        spec = S.CellularResourcePowerHistogram(
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


# %% Capture


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

    def test_tolerance_is_one_microsample(self):
        Capture(duration=(1000 + 1e-7) / 1e6, sample_rate=1e6)

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


# %% Cellular 5G NR SSB correlators


class TestSSBCorrelators:
    @by_correlator
    @given(kws=integer_delay_kwargs())
    @PROPERTY
    def test_integer_delay_samples_accepted(self, construct, cls, kws):
        spec = construct(cls, subcarrier_spacing=SCS, **kws)
        assert spec.delay == kws['delay']

    @by_correlator
    @given(kws=fractional_delay_kwargs())
    @PROPERTY
    def test_fractional_delay_samples_rejected(self, cls, kws):
        raises_on_both_paths(
            cls, msgspec.ValidationError, DELAY_MSG, subcarrier_spacing=SCS, **kws
        )

    def test_subcarrier_spacing_is_required(self):
        cls = CORRELATORS[0]
        with pytest.raises(TypeError, match="Missing required argument 'subcarrier"):
            cls()
        with pytest.raises(msgspec.ValidationError, match='subcarrier_spacing'):
            cls.from_dict({})

    @pytest.mark.parametrize(
        'cls, field, value, match',
        [
            (S.Cellular5GNRPSSCorrelator, 'max_lag_symbols', 7, r'<= 6'),
            (S.Cellular5GNRPSSCorrelator, 'max_lag_symbols', 0, r'>= 1'),
            (S.Cellular5GNPSSSync, 'window_fill', 0, r'> 0'),
            (S.Cellular5GNPSSSync, 'max_beams', 0, r'>= 1'),
        ],
        ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
    )
    def test_meta_bounds_enforced_only_on_convert(self, cls, field, value, match):
        # Meta constraints are checked by msgspec during conversion, not on
        # direct construction
        direct = cls(subcarrier_spacing=SCS, **{field: value})
        assert getattr(direct, field) == value

        with pytest.raises(msgspec.ValidationError, match=match):
            cls.from_dict({'subcarrier_spacing': SCS, field: value})
        with pytest.raises(msgspec.ValidationError, match=match):
            cls(subcarrier_spacing=SCS).replace(**{field: value})


# %% CellularCyclicAutocorrelator


class TestCellularCyclicAutocorrelator:
    @given(frame_range=valid_frame_ranges(), symbol_range=valid_symbol_ranges())
    @PROPERTY
    def test_valid_ranges_accepted(self, construct, frame_range, symbol_range):
        spec = construct(CCA, frame_range=frame_range, symbol_range=symbol_range)
        assert spec.frame_range == frame_range
        assert spec.symbol_range == symbol_range
        assert isinstance(spec.frame_range, tuple)
        assert isinstance(spec.symbol_range, tuple)

    @pytest.mark.parametrize('field', ['frame_range', 'symbol_range'])
    @given(range_=descending_ranges())
    @PROPERTY
    def test_descending_range_rejected(self, field, range_):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            f'{field} end must be >= start',
            **{field: range_},
        )

    def test_open_ended_symbol_range_needs_zero_start(self):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            'symbol_range end must be specified when start > 0',
            symbol_range=(1, None),
        )

    def test_open_ended_frame_range_fails_type_conversion(self):
        # the annotation forbids None, so conversion fails on type before
        # __post_init__ runs
        with pytest.raises(
            msgspec.ValidationError, match=r'Expected `int`, got `null`'
        ):
            CCA.from_dict({'frame_range': [1, None]})

    def test_zero_start_skips_range_check(self, construct):
        spec = construct(CCA, frame_range=(0, -5), symbol_range=(0, -1))
        assert spec.frame_range == (0, -5)
        assert spec.symbol_range == (0, -1)

    def test_int_range_form_skips_check(self, construct):
        spec = construct(CCA, frame_range=3, symbol_range=5)
        assert spec.frame_range == 3
        assert spec.symbol_range == 5
