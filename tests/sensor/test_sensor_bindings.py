"""striqt.sensor.lib.bindings: the tagged sweep union, schema binding of decoded
sweeps, BoundSweep validation, and out-of-tree binding registration"""

from __future__ import annotations

import re
from pathlib import Path

import msgspec
import pytest
from conftest import SWEEP_DIR
from sweep_strategies import SOURCE, SweepCls, make_sweep, sweep_dict

import striqt.sensor as ss

List = ss.specs.List
Repeat = ss.specs.Repeat
MOCK_MSG = 'no sensor was bound with this name'


# %% registry and tagged sweep type


def test_binding_introspect():
    union = ss.lib.bindings.get_tagged_sweep_type()
    msgspec.inspect.type_info(union)


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


def test_extension_module_registers_site_bindings():
    import site_strategies as S

    for name in ('site_single_tone', 'site_survey', 'site_single_tone_calibration'):
        assert name in ss.lib.bindings.registry
    assert Path(S.EXT.__file__).resolve().parent == (SWEEP_DIR / 'src').resolve()


def test_sensor_kwarg_typo_is_a_type_error():
    from site_strategies import EXT

    with pytest.raises(TypeError, match="unexpected keyword argument 'peripherals'"):
        ss.bindings.Sensor(
            source_cls=EXT.SiteToneSource, peripherals=ss.peripherals.NoPeripherals
        )


# %% BoundSweep.mock_source


class TestMockSource:
    def test_must_be_a_registered_binding(self):
        names = re.escape(repr(tuple(ss.lib.bindings.registry)))
        with pytest.raises(TypeError, match=MOCK_MSG) as excinfo:
            make_sweep(mock_source='bogus')
        assert re.search(names, str(excinfo.value))
        with pytest.raises(msgspec.ValidationError, match=MOCK_MSG):
            SweepCls.from_dict(sweep_dict(mock_source='bogus'))

    def test_check_runs_before_loop_validation(self):
        loops = (List(field='snr', values=(1.0,)), Repeat(count=2))
        with pytest.raises(TypeError, match=MOCK_MSG):
            make_sweep(mock_source='bogus', loops=loops)

    def test_registered_binding_accepted(self, construct):
        sweep = construct(SweepCls, source=SOURCE.to_dict(), mock_source='warmup')
        assert sweep.mock_source == 'warmup'

    @pytest.mark.xfail(
        strict=True,
        reason='BoundSweep.__post_init__ error text says mock_sensor, not mock_source',
    )
    def test_message_names_the_field(self):
        with pytest.raises(TypeError, match=r"mock_source 'bogus'"):
            make_sweep(mock_source='bogus')
