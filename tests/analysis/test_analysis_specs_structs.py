"""striqt.analysis.specs.structs: SpecBase freezing and __post_init__ validation"""

from __future__ import annotations

import math

import msgspec
import pytest
from analysis_strategies import (
    descending_ranges,
    fractional_delay_kwargs,
    fractional_sample_captures,
    integer_delay_kwargs,
    integer_sample_captures,
    valid_frame_ranges,
    valid_symbol_ranges,
)
from conftest import raises_on_both_paths
from hypothesis import given

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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            '_inspect_container_depth handles TupleType but not VarTupleType, so '
            'the list is never frozen and the lru-cached _validate cannot hash it'
        ),
    )
    def test_direct_struct_with_list_validates_and_replaces(self):
        spec = CCA(subcarrier_spacings=[15e3])
        assert spec.validate() == spec
        assert spec.replace(frame_range=(0, 2)).subcarrier_spacings == (15e3,)


# %% Capture


class TestCapture:
    @given(kws=integer_sample_captures())
    def test_integer_sample_count_is_accepted(self, construct, kws):
        capture = construct(Capture, **kws)
        assert capture.duration == kws['duration']
        assert capture.sample_rate == kws['sample_rate']

    @given(kws=fractional_sample_captures())
    def test_fractional_sample_count_is_rejected(self, kws):
        raises_on_both_paths(Capture, ValueError, PERIOD_MSG, **kws)

    def test_replace_revalidates_duration(self):
        capture = Capture(duration=1e-3, sample_rate=1e6)
        with pytest.raises(ValueError, match=PERIOD_MSG):
            capture.replace(duration=1.5e-6)

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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'types.SampleRate (analysis/specs/types.py:69) carries no gt=0 bound, '
            'unlike the sensor BackendSampleRate, so a negative rate converts'
        ),
    )
    def test_negative_sample_rate_is_rejected_on_convert(self):
        with pytest.raises(msgspec.ValidationError, match='sample_rate'):
            Capture.from_dict({'duration': 1e-3, 'sample_rate': -1e6})


# %% Cellular 5G NR SSB correlators


class TestSSBCorrelators:
    @by_correlator
    @given(kws=integer_delay_kwargs())
    def test_integer_delay_samples_accepted(self, construct, cls, kws):
        spec = construct(cls, subcarrier_spacing=SCS, **kws)
        assert spec.delay == kws['delay']

    @by_correlator
    @given(kws=fractional_delay_kwargs())
    def test_fractional_delay_samples_rejected(self, cls, kws):
        raises_on_both_paths(
            cls, msgspec.ValidationError, DELAY_MSG, subcarrier_spacing=SCS, **kws
        )

    def test_subcarrier_spacing_is_required(self):
        raises_on_both_paths(CORRELATORS[0], TypeError, 'subcarrier_spacing')

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
    def test_valid_ranges_accepted(self, construct, frame_range, symbol_range):
        spec = construct(CCA, frame_range=frame_range, symbol_range=symbol_range)
        assert spec.frame_range == frame_range
        assert spec.symbol_range == symbol_range
        assert isinstance(spec.frame_range, tuple)
        assert isinstance(spec.symbol_range, tuple)

    @pytest.mark.parametrize('field', ['frame_range', 'symbol_range'])
    @given(range_=descending_ranges())
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            '_validate_range (analysis/specs/structs.py:270) returns before the '
            'end >= start check when start is 0, so (0, -5) becomes an empty range'
        ),
    )
    @pytest.mark.parametrize('field', ['frame_range', 'symbol_range'])
    def test_descending_range_from_zero_rejected(self, field):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            f'{field} end must be >= start',
            **{field: (0, -5)},
        )
