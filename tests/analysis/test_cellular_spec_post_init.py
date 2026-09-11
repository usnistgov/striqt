"""__post_init__ validation of the cellular analysis specs"""

from __future__ import annotations

import msgspec
import pytest
from conftest import raises_on_both_paths
from hypothesis import HealthCheck, given, settings
from spec_strategies import (
    SSB_SAMPLE_RATE,
    descending_ranges,
    fractional_delay_kwargs,
    integer_delay_kwargs,
    valid_frame_ranges,
    valid_symbol_ranges,
)

import striqt.analysis as sa

S = sa.specs
CCA = S.CellularCyclicAutocorrelator
CORRELATORS = (
    S.Cellular5GNRPSSCorrelator,
    S.Cellular5GNRSSSCorrelator,
    S.Cellular5GNPSSSync,
    S.Cellular5GNSSSSync,
)
SCS = 30e3
DELAY_MSG = r'delay \* sample_rate must be an integer'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)

by_correlator = pytest.mark.parametrize('cls', CORRELATORS, ids=lambda c: c.__name__)


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

    @by_correlator
    def test_replace_revalidates_delay(self, cls):
        spec = cls(subcarrier_spacing=SCS)
        with pytest.raises(msgspec.ValidationError, match=DELAY_MSG):
            spec.replace(delay=1.5 / SSB_SAMPLE_RATE)

    @by_correlator
    def test_subcarrier_spacing_is_required(self, cls):
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


class TestCellularCyclicAutocorrelator:
    @given(frame_range=valid_frame_ranges(), symbol_range=valid_symbol_ranges())
    @PROPERTY
    def test_valid_ranges_accepted(self, construct, frame_range, symbol_range):
        spec = construct(CCA, frame_range=frame_range, symbol_range=symbol_range)
        assert spec.frame_range == frame_range
        assert spec.symbol_range == symbol_range
        assert isinstance(spec.frame_range, tuple)
        assert isinstance(spec.symbol_range, tuple)

    @given(range_=descending_ranges())
    @PROPERTY
    def test_descending_frame_range_rejected(self, range_):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            'frame_range end must be >= start',
            frame_range=range_,
        )

    @given(range_=descending_ranges())
    @PROPERTY
    def test_descending_symbol_range_rejected(self, range_):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            'symbol_range end must be >= start',
            symbol_range=range_,
        )

    def test_open_ended_symbol_range_needs_zero_start(self):
        raises_on_both_paths(
            CCA,
            msgspec.ValidationError,
            'symbol_range end must be specified when start > 0',
            symbol_range=(1, None),
        )

    def test_open_ended_frame_range_with_positive_start(self):
        with pytest.raises(msgspec.ValidationError, match='frame_range end must be'):
            CCA(frame_range=(1, None))
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

    def test_replace_revalidates_range(self):
        with pytest.raises(msgspec.ValidationError, match='frame_range end must be'):
            CCA().replace(frame_range=(2, 1))
