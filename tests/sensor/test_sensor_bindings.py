"""striqt.sensor.lib.bindings: the tagged sweep union, schema binding of decoded
sweeps, BoundSweep validation, and out-of-tree binding registration"""

from __future__ import annotations

import re
from pathlib import Path

import msgspec
import pytest
from conftest import SWEEP_DIR, raises_on_both_paths
from sweep_strategies import SweepCls, make_sweep, make_sweep_kws

import striqt.sensor as ss

MOCK_MSG = 'no sensor was bound with this name'


# %% registry and tagged sweep type


def test_tagged_union_arms_match_the_registry():
    info = msgspec.inspect.type_info(ss.lib.bindings.get_tagged_sweep_type())
    registry = ss.lib.bindings.registry
    assert {t.tag for t in info.types} == set(registry)
    for t in info.types:
        assert registry[t.tag].sensor.sweep_spec_cls is t.cls


def test_yaml_schema_binding(spec_dir):
    spec = ss.read_yaml_spec(spec_dir / 'air7101b.yaml')
    ctrl_cls = ss.bindings.air7101b
    schema = ctrl_cls.schema

    assert isinstance(spec.captures[0], schema.capture), 'capture binding mismatch'
    assert isinstance(spec.peripherals, schema.peripherals), 'periph binding mismatch'
    assert isinstance(spec.source, schema.source), 'source binding mismatch'
    assert isinstance(spec, ctrl_cls.sensor.sweep_spec_cls), (
        'sweep spec binding mismatch'
    )


def test_extension_module_registers_site_bindings(subtests):
    import site_strategies as S

    for name in ('site_single_tone', 'site_survey', 'site_single_tone_calibration'):
        with subtests.test(msg=name):
            assert name in ss.lib.bindings.registry
    assert Path(S.EXT.__file__).resolve().parent == (SWEEP_DIR / 'src').resolve()


# %% BoundSweep.mock_source


class TestMockSource:
    def test_mock_source_must_be_a_registered_binding(self):
        names = re.escape(repr(tuple(ss.lib.bindings.registry)))
        raises_on_both_paths(
            SweepCls,
            TypeError,
            f'{MOCK_MSG}.*{names}',
            **make_sweep_kws(mock_source='bogus'),
        )
        assert make_sweep(mock_source='warmup').mock_source == 'warmup'

    @pytest.mark.xfail(
        strict=True,
        reason='BoundSweep.__post_init__ error text says mock_sensor, not mock_source',
    )
    def test_message_names_the_field(self):
        with pytest.raises(TypeError, match=r"mock_source 'bogus'"):
            make_sweep(mock_source='bogus')
