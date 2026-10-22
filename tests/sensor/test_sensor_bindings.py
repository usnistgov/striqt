"""striqt.sensor.lib.bindings: the tagged sweep union and binding registration"""

from __future__ import annotations

import dataclasses
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


def test_bound_sweep_names_its_controller_and_sweep_class():
    """the tagged class of a plain sweep binding is its own `sweep_cls`, and its
    `controller` is the registered controller class"""
    spec = ss.read_yaml_spec(SWEEP_DIR / 'air7101b.yaml')
    assert type(spec).controller is ss.bindings.air7101b
    assert type(spec).sweep_cls is type(spec)
    assert type(spec).__mro__.index(ss.specs.Sweep) == 1, 'one bound class, no nesting'


def test_bound_sweep_and_controller_share_one_binding():
    """`type(spec).sensor` is the `SensorBinding` object of `get_controller(spec)`,
    and its `sweep_spec_cls` is the tagged class"""
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
            tone_schema(action=ss.specs.ManualYFactorCalibration),
            register=False,
        )
        sweep_cls = ctrl_cls.sensor.sweep_spec_cls
        assert issubclass(sweep_cls, ss.specs.ManualYFactorCalibration)
        assert sweep_cls.__name__ == 'action_test'
        assert 'calibration' in sweep_cls.__struct_fields__

    def test_a_non_sweep_action_gets_a_companion_sweep_class(self, isolated_bindings):
        """the companion class has the action's spec types and binding and is neither
        tagged nor registered. The runner builds the sweep that runs from it."""
        schema = tone_schema(action=ss.specs.ManualYFactorCalibration)
        ctrl_cls = ss.bindings.bind_sensor(
            'companion_test', tone_sensor(), schema, register=False
        )
        action_cls = ctrl_cls.sensor.sweep_spec_cls
        companion = action_cls.sweep_cls
        assert companion is not action_cls
        assert issubclass(companion, ss.specs.Sweep)
        assert companion.__name__ == 'companion_test.sweep'
        assert companion.schema is schema
        assert companion.sensor is ctrl_cls.sensor
        assert companion.controller is ctrl_cls
        assert companion.sweep_cls is companion
        assert 'companion_test.sweep' not in isolated_bindings.registry
        assert 'mock_source' in companion.__struct_fields__

    def test_sweep_spec_cls_still_selects_the_sweep_base(self, isolated_bindings):
        ctrl_cls = ss.bindings.bind_sensor(
            'legacy_test',
            tone_sensor(sweep_spec_cls=ss.specs.ManualYFactorCalibration),
            tone_schema(),
            register=False,
        )
        assert issubclass(
            ctrl_cls.sensor.sweep_spec_cls, ss.specs.ManualYFactorCalibration
        )

    def test_an_existing_binding_derives_through_schema_action(self, isolated_bindings):
        """the `sweep_spec_cls` of a `SensorBinding` is its tagged class, which is
        not a base for a second binding. The action comes from the schema."""
        ctrl_cls = ss.bindings.bind_sensor(
            'derived_test',
            TONE.sensor,
            tone_schema(action=ss.specs.ManualYFactorCalibration),
            register=False,
        )
        sweep_cls = ctrl_cls.sensor.sweep_spec_cls
        assert issubclass(sweep_cls, ss.specs.ManualYFactorCalibration)
        assert not issubclass(sweep_cls, TONE.sensor.sweep_spec_cls)

    def test_conflicting_action_types_are_rejected(self, isolated_bindings):
        with pytest.raises(TypeError, match=r'sweep_spec_cls.*schema\.action'):
            ss.bindings.bind_sensor(
                'conflict_test',
                tone_sensor(sweep_spec_cls=ss.specs.ManualYFactorCalibration),
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
        """`get_controller` finds registered bindings only. A file that decoded into
        an unregistered one could not be run."""
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
    """the extension module binds its calibration with `bind_sensor`, and the API
    documentation groups the controller under that module"""
    import site_strategies as S

    assert S.EXT.site_single_tone_calibration.__module__ == S.EXT.__name__
    assert S.EXT.site_single_tone.__module__ == S.EXT.__name__


# %% bind_manual_yfactor_calibration


def test_bind_manual_yfactor_calibration_is_deprecated(isolated_bindings):
    """the function warns and binds what `bind_sensor` binds from the schema with
    `action` replaced by `ManualYFactorCalibration`"""
    with pytest.warns(DeprecationWarning, match='bind_sensor'):
        derived = ss.lib.bindings.bind_manual_yfactor_calibration(
            'deprecated_cal_test', TONE
        )
    direct = ss.bindings.bind_sensor(
        'direct_cal_test',
        TONE.sensor,
        dataclasses.replace(TONE.schema, action=ss.specs.ManualYFactorCalibration),
        register=False,
    )
    assert derived.schema == direct.schema
    assert derived.schema.action is ss.specs.ManualYFactorCalibration
    for ctrl_cls in (derived, direct):
        action_cls = ctrl_cls.sensor.sweep_spec_cls
        assert action_cls.__mro__[1] is ss.specs.ManualYFactorCalibration
        assert action_cls.sweep_cls.__mro__[1] is ss.specs.Sweep
    assert (
        derived.sensor.sweep_spec_cls.__struct_fields__
        == direct.sensor.sweep_spec_cls.__struct_fields__
    )
    assert derived.__module__ == __name__
    assert isolated_bindings.registry['deprecated_cal_test'] is derived


# %% cellular bindings and ActionSpec.check_schema

CELLULAR = ss.bindings.air7101b_cellular


class TestCellularBinding:
    def test_tagged_class_subclasses_the_cellular_sweep(self):
        tagged_cls = CELLULAR.sensor.sweep_spec_cls
        assert tagged_cls.__mro__.index(ss.specs.Cellular5GNRSweep) == 1
        assert tagged_cls.__name__ == 'air7101b_cellular'
        assert 'cell_channels' in tagged_cls.__struct_fields__

    def test_companion_sweep_class(self):
        sweep_cls = CELLULAR.sensor.sweep_spec_cls.sweep_cls
        assert sweep_cls.__mro__.index(ss.specs.Sweep) == 1
        assert sweep_cls.__name__ == 'air7101b_cellular.sweep'
        assert sweep_cls.controller is CELLULAR

    def test_runner_is_the_cellular_runner(self):
        runner_cls = ss.lib.actions.registry.lookup(CELLULAR.sensor.sweep_spec_cls)
        assert runner_cls is ss.lib.actions.Cellular5GNRSweepRunner

    def test_registered_under_its_key(self):
        assert ss.lib.bindings.get_controller('air7101b_cellular') is CELLULAR

    @pytest.mark.parametrize('name', ['air7101b', 'air7201b', 'air8201b'])
    def test_cellular_binding_shares_the_sensor_of_its_origin(self, name):
        origin = getattr(ss.bindings, name)
        cellular = getattr(ss.bindings, f'{name}_cellular')
        assert cellular.sensor.source_cls is origin.sensor.source_cls
        assert cellular.schema == dataclasses.replace(
            origin.schema, action=ss.specs.Cellular5GNRSweep
        )


class TestCheckSchema:
    def test_cellular_action_needs_a_center_frequency_field(self, isolated_bindings):
        """`SingleToneCapture` has no `center_frequency`, which `cell_channels` is
        matched against"""
        assert 'center_frequency' not in TONE.schema.capture.__struct_fields__
        with pytest.raises(TypeError, match='center_frequency'):
            ss.bindings.bind_sensor(
                'probe',
                tone_sensor(),
                tone_schema(action=ss.specs.Cellular5GNRSweep),
                register=False,
            )

    def test_cellular_action_accepts_a_capture_with_center_frequency(self):
        assert ss.specs.Cellular5GNRSweep.check_schema(CELLULAR.schema) is None

    @pytest.mark.parametrize(
        'action',
        [ss.specs.ActionSpec, ss.specs.Sweep, ss.specs.ManualYFactorCalibration],
    )
    def test_base_actions_accept_any_schema(self, action):
        for schema in (tone_schema(), CELLULAR.schema):
            assert action.check_schema(schema) is None


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
