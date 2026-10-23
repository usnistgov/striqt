"""striqt.sensor.lib.bindings: the tagged sweep union and binding registration"""

# ruff: noqa: UP045  msgspec resolves the analysis group annotations at runtime on py3.9
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

import msgspec
import numpy as np
import pytest
from conftest import SWEEP_DIR, construct_both, raises_on_both_paths
from sweep_strategies import (
    SOURCE,
    CellularAnalyses,
    QuickAnalyses,
    SweepCls,
    make_capture,
    make_sweep,
    make_sweep_kws,
    quick_binding,
)
from synthetic_sources import run_in_memory

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.helpers import SpecValidationError

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


def test_yaml_spec_decodes_into_the_tagged_class_of_its_binding(subtests):
    """the tagged class of a plain sweep binding is its own `sweep_cls`, its
    `controller` is the registered controller class, and its `sensor` is the
    `SensorBinding` of `get_controller(spec)`"""
    spec = ss.read_yaml_spec(SWEEP_DIR / 'air7101b.yaml')
    ctrl_cls = ss.bindings.air7101b
    cls = type(spec)

    with subtests.test(msg='schema types'):
        assert isinstance(spec.captures[0], ctrl_cls.schema.capture)
        assert isinstance(spec.peripherals, ctrl_cls.schema.peripherals)
        assert isinstance(spec.source, ctrl_cls.schema.source)
    with subtests.test(msg='tagged class'):
        assert cls is ctrl_cls.sensor.sweep_spec_cls
        assert cls.__mro__.index(ss.specs.Sweep) == 1, 'one bound class, no nesting'
        assert cls.sweep_cls is cls
    with subtests.test(msg='controller and binding'):
        assert ss.lib.bindings.get_controller(spec) is ctrl_cls
        assert cls.controller is ctrl_cls
        assert cls.sensor is ctrl_cls.sensor


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
    `action` replaced by `ManualYFactorCalibration`. The controller's module is the
    caller's unless `module=` names another."""
    with pytest.warns(DeprecationWarning, match='bind_sensor'):
        derived = ss.lib.bindings.bind_manual_yfactor_calibration(
            'deprecated_cal_test', TONE
        )
    direct = ss.bindings.bind_sensor(
        'direct_cal_test',
        TONE.sensor,
        dataclasses.replace(TONE.schema, action=ss.specs.ManualYFactorCalibration),
        register=False,
        module='some.extension',
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
    assert direct.__module__ == 'some.extension'
    assert isolated_bindings.registry['deprecated_cal_test'] is derived


# %% cellular bindings and ActionSpec.check_schema

CELLULAR = ss.bindings.air7101b_cellular


class TestCellularBinding:
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

    def test_calibration_action_needs_channel_power_time_series(self):
        """the calibration generates a `channel_power_time_series` analysis"""
        with pytest.raises(TypeError, match='channel_power_time_series'):
            ss.bindings.bind_sensor(
                'probe',
                tone_sensor(),
                tone_schema(
                    action=ss.specs.ManualYFactorCalibration, analysis=SpectrogramOnly
                ),
                register=False,
            )

    def test_cellular_action_needs_every_measurement_it_generates(self):
        """`QuickAnalyses` lacks `power_spectral_density`, the second field of
        `Cellular5GNRAnalysis`"""
        schema = dataclasses.replace(CELLULAR.schema, analysis=QuickAnalyses)
        with pytest.raises(TypeError, match='power_spectral_density'):
            ss.specs.Cellular5GNRSweep.check_schema(schema)

    def test_cellular_action_accepts_a_group_of_its_measurements(self):
        schema = dataclasses.replace(CELLULAR.schema, analysis=CellularAnalyses)
        assert ss.specs.Cellular5GNRSweep.check_schema(schema) is None


# %% Schema.analysis and the bound analysis field


class SpectrogramOnly(
    sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    spectrogram: Optional[sa.specs.Spectrogram] = None


QUICK_ANALYSIS = {
    'channel_power_time_series': {'detector_period': 1e-4, 'power_detectors': ['rms']}
}
PSD_ANALYSIS = {
    'power_spectral_density': {'frequency_resolution': 1e4, 'window': 'hann'}
}


def analysis_field(cls) -> msgspec.structs.FieldInfo:
    return {f.name: f for f in msgspec.structs.fields(cls)}['analysis']


class TestAnalysisSchema:
    def test_default_group_is_the_builtin_analyses(self):
        assert tone_schema().analysis is ss.specs.BuiltinAnalyses
        assert analysis_field(SweepCls).type is ss.specs.BuiltinAnalyses

    def test_schema_rejects_a_type_that_is_not_an_analysis_group(self):
        with pytest.raises(AssertionError):
            tone_schema(analysis=ss.specs.Peripherals)

    def test_bound_sweep_annotates_analysis_with_the_group(self):
        """the field is re-annotated in place: same field order, one bound class"""
        cls = quick_binding().sensor.sweep_spec_cls
        field = analysis_field(cls)
        assert field.type is QuickAnalyses
        assert field.default_factory is QuickAnalyses
        assert cls.__struct_fields__ == SweepCls.__struct_fields__
        assert cls.__mro__.index(ss.specs.Sweep) == 1

    def test_measurement_in_the_group_decodes_as_the_group(self):
        cls = quick_binding().sensor.sweep_spec_cls
        kws = make_sweep_kws(
            captures=(make_capture(),), analysis=QuickAnalyses.from_dict(QUICK_ANALYSIS)
        )
        for sweep in construct_both(cls, **kws):
            assert type(sweep.analysis) is QuickAnalyses
            assert set(sweep.analysis.to_dict()) == {'channel_power_time_series'}
            period = sweep.analysis.channel_power_time_series.detector_period
            assert period == pytest.approx(1e-4)

    def test_measurement_outside_the_group_is_rejected(self):
        """a `BuiltinAnalyses` passed directly, or a dict naming a measurement the
        group lacks, both fail at `analysis`"""
        cls = quick_binding().sensor.sweep_spec_cls
        raises_on_both_paths(
            cls,
            SpecValidationError,
            r'QuickAnalyses|power_spectral_density',
            **make_sweep_kws(
                captures=(make_capture(),),
                analysis=ss.specs.BuiltinAnalyses.from_dict(PSD_ANALYSIS),
            ),
        )

    def test_bind_sensor_rejects_an_unregistered_spec_type(self):
        class Unregistered(
            sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
        ):
            capture: Optional[sa.specs.Capture] = None

        with pytest.raises(TypeError, match='Capture is not a registered'):
            ss.bindings.bind_sensor(
                'probe',
                tone_sensor(),
                tone_schema(analysis=Unregistered),
                register=False,
            )

    def test_bind_sensor_rejects_a_field_named_unlike_its_measurement(self):
        class Renamed(
            sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
        ):
            psd: Optional[sa.specs.PowerSpectralDensity] = None

        with pytest.raises(
            TypeError, match="psd must be named 'power_spectral_density'"
        ):
            ss.bindings.bind_sensor(
                'probe', tone_sensor(), tone_schema(analysis=Renamed), register=False
            )

    def test_mock_binding_keeps_the_group_of_its_origin(self):
        mock = ss.lib.bindings.mock_binding(
            quick_binding(), ss.bindings.noise, register=False
        )
        assert mock.schema.analysis is QuickAnalyses
        assert analysis_field(mock.sensor.sweep_spec_cls).type is QuickAnalyses

    def test_calibration_companion_sweep_carries_the_group(self):
        cal = quick_binding(ss.specs.ManualYFactorCalibration)
        assert analysis_field(cal.sensor.sweep_spec_cls.sweep_cls).type is QuickAnalyses


# %% extension measurements


class ToyPower(sa.specs.Analysis, frozen=True, kw_only=True):
    scale: float = 1.0


def test_an_extension_measurement_runs_from_the_analysis_block(
    isolated_analysis_registry, isolated_bindings
):
    """a measurement registered outside striqt.analysis.measurements is reached from
    the ``analysis:`` block of a binding whose group names it, and the value of the
    spec field in the block reaches the output variable"""

    @sa.registry.measurement(ToyPower, dtype='float32', dims=())
    def toy_power(iq, capture, scale: float = 1.0):
        return np.full(iq.shape[0], scale, dtype='float32')

    class ToyAnalyses(
        sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
    ):
        toy_power: Optional[ToyPower] = None
        channel_power_time_series: Optional[sa.specs.ChannelPowerTimeSeries] = None

    ctrl = ss.bindings.bind_sensor(
        'toy_tone', TONE.sensor, dataclasses.replace(TONE.schema, analysis=ToyAnalyses)
    )
    tree = {
        'source': SOURCE.to_dict(),
        'captures': [make_capture(port=(0, 1)).to_dict()],
        'analysis': {'toy_power': {'scale': 3.0}},
        'sink': {'path': None},
    }
    sweep = ctrl.sensor.sweep_spec_cls.from_dict(tree)
    assert type(sweep.analysis) is ToyAnalyses

    (ds,) = run_in_memory(sweep)
    assert ds['toy_power'].values.tolist() == [3.0, 3.0]
    assert set(sweep.analysis.to_dict()) == {'toy_power'} <= set(ds.data_vars)


# %% BoundSweep.mock_source


class TestMockSource:
    def test_mock_source_must_be_a_registered_binding(self):
        names = re.escape(repr(tuple(ss.lib.bindings.registry)))
        raises_on_both_paths(
            SweepCls,
            TypeError,
            f"mock_source 'bogus': {MOCK_MSG}.*{names}",
            **make_sweep_kws(mock_source='bogus'),
        )
        assert make_sweep(mock_source='warmup').mock_source == 'warmup'
