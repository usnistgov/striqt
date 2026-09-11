"""__post_init__ validation of SoapySource and its hardware subclasses"""

from __future__ import annotations

import msgspec
import pytest
from conftest import raises_on_both_paths
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import striqt.analysis as sa
import striqt.sensor as ss

SoapySource = ss.specs.SoapySource
Air7101BSource = ss.bindings.air7101b.schema.source
MCR = {'master_clock_rate': 1e6}
TRIGGERS = ('cellular_5g_pss_sync', 'cellular_5g_sss_sync')
SYNC_MSG = 'time_sync_at must be "open" when gapless'
RETRIES_MSG = 'receive_retries must be 0 when gapless is enabled'
TRIGGER_MSG = 'signal_trigger must be one of'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


class TestGapless:
    @pytest.mark.parametrize('time_sync_at', ['open', 'acquire'])
    @pytest.mark.parametrize('receive_retries', [0, 3])
    def test_non_gapless_ignores_sync_and_retries(
        self, construct, time_sync_at, receive_retries
    ):
        source = construct(
            SoapySource,
            gapless=False,
            time_sync_at=time_sync_at,
            receive_retries=receive_retries,
            **MCR,
        )
        assert source.time_sync_at == time_sync_at
        assert source.receive_retries == receive_retries

    @pytest.mark.parametrize('kws', [{}, {'time_sync_at': 'acquire'}])
    def test_gapless_requires_open_time_sync(self, kws):
        raises_on_both_paths(
            SoapySource, ValueError, SYNC_MSG, gapless=True, **kws, **MCR
        )

    @pytest.mark.parametrize('receive_retries', [1, 3])
    def test_gapless_requires_zero_retries(self, receive_retries):
        raises_on_both_paths(
            SoapySource,
            ValueError,
            RETRIES_MSG,
            gapless=True,
            time_sync_at='open',
            receive_retries=receive_retries,
            **MCR,
        )

    def test_gapless_open_zero_retries_ok(self, construct):
        source = construct(SoapySource, gapless=True, time_sync_at='open', **MCR)
        assert source.gapless is True

    def test_sync_check_precedes_retries_check(self):
        with pytest.raises(ValueError, match=SYNC_MSG):
            SoapySource(gapless=True, time_sync_at='acquire', receive_retries=3, **MCR)

    def test_replace_to_gapless_revalidates(self):
        with pytest.raises(ValueError, match=SYNC_MSG):
            SoapySource(**MCR).replace(gapless=True)

    def test_air7101b_default_retries_forbid_gapless(self):
        assert Air7101BSource().receive_retries == 3
        with pytest.raises(ValueError, match=RETRIES_MSG):
            Air7101BSource(gapless=True, time_sync_at='open')
        assert Air7101BSource(gapless=True, time_sync_at='open', receive_retries=0)


class TestSignalTrigger:
    @pytest.mark.parametrize('name', [*TRIGGERS, None])
    def test_registered_names_accepted(self, construct, name):
        source = construct(SoapySource, signal_trigger=name, **MCR)
        assert source.signal_trigger == name

    @given(
        name=st.text(min_size=1).filter(lambda s: s not in sa.registry.signal_trigger)
    )
    @PROPERTY
    def test_unregistered_name_rejected(self, name):
        raises_on_both_paths(
            SoapySource, ValueError, TRIGGER_MSG, signal_trigger=name, **MCR
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'signal_trigger is annotated Union[str, AnalysisGroup, None] but '
            '__post_init__ only accepts registered trigger names'
        ),
    )
    def test_analysis_group_accepted(self):
        # BundledTriggers() cannot be instantiated on py3.14, so the bare base is used
        group = sa.specs.AnalysisGroup()
        source = SoapySource(signal_trigger=group, **MCR)
        assert isinstance(source.signal_trigger, sa.specs.AnalysisGroup)


def test_time_sync_at_literal_enforced_only_on_convert():
    assert SoapySource(time_sync_at='never', **MCR).time_sync_at == 'never'
    with pytest.raises(msgspec.ValidationError, match="Invalid enum value 'never'"):
        SoapySource.from_dict(dict(time_sync_at='never', **MCR))
