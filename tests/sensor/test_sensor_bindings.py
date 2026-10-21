"""striqt.sensor.lib.bindings: the tagged sweep union and binding registration"""

from __future__ import annotations

import re
from pathlib import Path

import msgspec
import pytest
from conftest import SWEEP_DIR, raises_on_both_paths
from sweep_strategies import SweepCls, make_sweep, make_sweep_kws

import striqt.sensor as ss

MOCK_MSG = 'no sensor was bound with this name'

TONE = ss.bindings.single_tone


def tone_sensor(**kws) -> ss.bindings.Sensor:
    return ss.bindings.Sensor(source_cls=TONE.sensor.source_cls, **kws)


def tone_schema(**kws) -> ss.bindings.Schema:
    return ss.bindings.Schema(
        source=TONE.schema.source,
        init_like=TONE.schema.source,
        capture=TONE.schema.capture,
        arm_like=TONE.schema.capture,
        peripherals=TONE.schema.peripherals,
        **kws,
    )


# %% registry and tagged sweep type


def test_tagged_union_arms_match_the_registry():
    info = msgspec.inspect.type_info(ss.lib.bindings.get_tagged_sweep_type())
    registry = ss.lib.bindings.registry
    assert {t.tag for t in info.types} == set(registry)
    for t in info.types:
        assert registry[t.tag].sensor.sweep_spec_cls is t.cls


def test_yaml_schema_binding():
    spec = ss.read_yaml_spec(SWEEP_DIR / 'air7101b.yaml')
    ctrl_cls = ss.bindings.air7101b
    schema = ctrl_cls.schema

    assert isinstance(spec.captures[0], schema.capture), 'capture binding mismatch'
    assert isinstance(spec.peripherals, schema.peripherals), 'periph binding mismatch'
    assert isinstance(spec.source, schema.source), 'source binding mismatch'
    assert isinstance(spec, ctrl_cls.sensor.sweep_spec_cls), (
        'sweep spec binding mismatch'
    )


def test_bound_sweep_and_controller_share_one_binding():
    """`get_controller(spec)` and `type(spec).sensor` must describe the same binding,
    so the class attribute is the controller's `SensorBinding` and names the tagged
    class itself"""
    spec = ss.read_yaml_spec(SWEEP_DIR / 'air7101b.yaml')
    ctrl_cls = ss.lib.bindings.get_controller(spec)
    assert type(spec).sensor is ctrl_cls.sensor
    assert ctrl_cls.sensor.sweep_spec_cls is type(spec)


# %% bind_sensor: action selection and module


class TestBindSensor:
    def test_schema_action_selects_the_sweep_base(self, isolated_bindings):
        ctrl_cls = ss.bindings.bind_sensor(
            'action_test',
            tone_sensor(),
            tone_schema(action=ss.specs.CalibrationSweep),
            register=False,
        )
        sweep_cls = ctrl_cls.sensor.sweep_spec_cls
        assert issubclass(sweep_cls, ss.specs.CalibrationSweep)
        assert sweep_cls.__name__ == 'action_test'
        assert 'calibration' in sweep_cls.__struct_fields__

    def test_sweep_spec_cls_still_selects_the_sweep_base(self, isolated_bindings):
        ctrl_cls = ss.bindings.bind_sensor(
            'legacy_test',
            tone_sensor(sweep_spec_cls=ss.specs.CalibrationSweep),
            tone_schema(),
            register=False,
        )
        assert issubclass(ctrl_cls.sensor.sweep_spec_cls, ss.specs.CalibrationSweep)

    def test_an_existing_binding_derives_through_schema_action(self, isolated_bindings):
        """a `SensorBinding` names its own tagged class, which is not a base for a
        second binding, so the action comes from the schema"""
        ctrl_cls = ss.bindings.bind_sensor(
            'derived_test',
            TONE.sensor,
            tone_schema(action=ss.specs.CalibrationSweep),
            register=False,
        )
        sweep_cls = ctrl_cls.sensor.sweep_spec_cls
        assert issubclass(sweep_cls, ss.specs.CalibrationSweep)
        assert not issubclass(sweep_cls, TONE.sensor.sweep_spec_cls)

    def test_conflicting_action_types_are_rejected(self, isolated_bindings):
        with pytest.raises(TypeError, match=r'sweep_spec_cls.*schema\.action'):
            ss.bindings.bind_sensor(
                'conflict_test',
                tone_sensor(sweep_spec_cls=ss.specs.CalibrationSweep),
                tone_schema(action=SweepCls),
                register=False,
            )

    def test_module_defaults_to_the_caller(self, isolated_bindings):
        ctrl_cls = ss.bindings.bind_sensor(
            'module_default_test', tone_sensor(), tone_schema(), register=False
        )
        assert ctrl_cls.__module__ == __name__

    def test_module_argument_is_recorded(self, isolated_bindings):
        ctrl_cls = ss.bindings.bind_sensor(
            'module_test',
            tone_sensor(),
            tone_schema(),
            register=False,
            module='some.extension',
        )
        assert ctrl_cls.__module__ == 'some.extension'

    def test_unregistered_binding_stays_out_of_the_registry(self, isolated_bindings):
        ss.bindings.bind_sensor(
            'unregistered_test', tone_sensor(), tone_schema(), register=False
        )
        assert 'unregistered_test' not in isolated_bindings.registry

    def test_unregistered_binding_stays_out_of_the_union(self, isolated_bindings):
        """`get_controller` cannot find an unregistered binding, so a file that
        decodes into one could not be run"""
        union_before = isolated_bindings.get_tagged_sweep_type()
        ss.bindings.bind_sensor(
            'unregistered_test', tone_sensor(), tone_schema(), register=False
        )
        assert isolated_bindings.get_tagged_sweep_type() is union_before


def test_extension_module_registers_site_bindings(subtests):
    import site_strategies as S

    for name in ('site_single_tone', 'site_survey', 'site_single_tone_calibration'):
        with subtests.test(msg=name):
            assert name in ss.lib.bindings.registry
    assert Path(S.EXT.__file__).resolve().parent == (SWEEP_DIR / 'src').resolve()


def test_derived_calibration_binding_records_the_extension_module():
    """`bind_manual_yfactor_calibration` is called from the extension module, which
    is where the API documentation must place the controller"""
    import site_strategies as S

    assert S.EXT.site_single_tone_calibration.__module__ == S.EXT.__name__
    assert S.EXT.site_single_tone.__module__ == S.EXT.__name__


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

    def test_message_names_the_field(self):
        with pytest.raises(TypeError, match=r"mock_source 'bogus'"):
            make_sweep(mock_source='bogus')
