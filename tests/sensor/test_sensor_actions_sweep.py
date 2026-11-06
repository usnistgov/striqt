"""striqt.sensor.lib.actions.sweep: SweepRunner, and the iterate_sweep dispatch through
resources['runner'].

The pipeline run by `SweepRunner.iterate` is covered in `test_sensor_actions_core.py`
through `ss.iterate_sweep`.
"""

from __future__ import annotations

import pytest
from sweep_strategies import SweepCls
from synthetic_sources import tone_sweep

import striqt.sensor as ss
from striqt.sensor.lib.actions import SweepRunner, registry

OFFSETS = (1e6, 2e6)


# %% SweepRunner


def test_a_bound_sweep_dispatches_to_the_sweep_runner():
    assert registry.lookup(tone_sweep(OFFSETS)) is SweepRunner


def test_build_sweep_is_the_spec_itself():
    sweep = tone_sweep(OFFSETS)
    runner = SweepRunner(sweep)
    assert runner.sweep is sweep
    assert runner.spec is sweep
    with ss.open_resources(sweep, None) as res:
        assert res['sweep_spec'] is sweep
        assert res['action_spec'] is sweep
        assert isinstance(res['runner'], SweepRunner)


# %% iterate_sweep dispatch


class RecordingRunner(SweepRunner):
    """a `SweepRunner` with an `iterate` that yields the received resources and
    keyword arguments"""

    def iterate(
        self,
        resources,
        *,
        always_yield=False,
        yield_values=True,
        loop=False,
        captures=None,
    ):
        yield (
            resources,
            {
                'always_yield': always_yield,
                'yield_values': yield_values,
                'loop': loop,
                'captures': captures,
            },
        )


def test_iterate_sweep_replaces_resources_in_a_copy():
    sweep = tone_sweep(OFFSETS)
    runner = RecordingRunner(sweep)
    res = ss.lib.resources.Resources(
        action_spec=sweep, sweep_spec=sweep, runner=runner, format_path=None
    )
    token = object()

    ((received, _),) = list(ss.iterate_sweep(res, calibration=token))

    assert received['calibration'] is token
    assert 'calibration' not in res


def test_open_resources_builds_the_registered_runner(monkeypatch):
    monkeypatch.setitem(registry, ss.specs.Sweep, RecordingRunner)
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None) as res:
        assert type(res['runner']) is RecordingRunner
        assert next(iter(ss.iterate_sweep(res)))[0] == res


def test_open_resources_raises_for_an_unregistered_spec_type(monkeypatch):
    """with the Sweep registration removed, nothing in the MRO of a bound sweep
    is registered"""
    monkeypatch.delitem(registry, ss.specs.Sweep)
    with pytest.raises(TypeError, match=SweepCls.__name__):
        ss.open_resources(tone_sweep(OFFSETS), None)
