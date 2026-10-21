"""striqt.sensor.lib.actions.sweep: SweepRunner, and the iterate_sweep dispatch through
resources['runner'].

The runs use 1-port scale-only tones at distinct offsets. A result attributed to
the wrong capture then disagrees with its generator oracle exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from sweep_strategies import SweepCls
from synthetic_sources import expected_corrected, tone_captures, tone_sweep

import striqt.sensor as ss
from striqt.sensor.lib.actions import SweepRunner, registry

OFFSETS = (1e6, 2e6)


def materialize(items) -> list:
    """the xarray datasets of the non-None items"""
    return [ss.lib.compute.from_delayed(dd) for dd in items if dd is not None]


# %% SweepRunner


@pytest.mark.parametrize(
    'spec',
    [tone_sweep(OFFSETS), SweepCls],
    ids=['instance', 'type'],
)
def test_a_bound_sweep_dispatches_to_the_sweep_runner(spec):
    assert registry.lookup(spec) is SweepRunner


def test_sweep_runner_matches_the_pipeline_and_the_oracle(subtests):
    """`iterate` with its defaults yields what `iterate_captures` does: 2 leading
    None placeholders, then one sink result per capture, each the tone the
    generator oracle predicts for that capture"""
    captures = tone_captures(OFFSETS)
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None) as res:
        items = list(SweepRunner(sweep).iterate(res))

    assert [item is None for item in items] == [True, True, False, False]
    datasets = materialize(items)
    for i, (capture, ds) in enumerate(zip(captures, datasets)):
        with subtests.test(capture=i):
            assert int(ds.capture_index.values[0]) == i
            expected = expected_corrected('single_tone', capture)
            assert np.array_equal(ds.iq_waveform.values, expected)


def test_sweep_runner_forwards_the_yield_modes():
    """placeholders only: one None per pipeline stage (N captures + 2)"""
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None) as res:
        runner = SweepRunner(sweep)
        items = list(runner.iterate(res, yield_values=False, always_yield=True))
    assert items == [None] * (len(OFFSETS) + 2)


def test_build_sweep_is_the_spec_itself():
    sweep = tone_sweep(OFFSETS)
    runner = SweepRunner(sweep)
    assert runner.sweep is sweep
    assert runner.spec is sweep
    with ss.open_resources(sweep, None) as res:
        assert res['sweep_spec'] is sweep
        assert res['action_spec'] is sweep
        assert isinstance(res['runner'], SweepRunner)


def test_validate_accepts_every_sweep():
    sweep = tone_sweep(OFFSETS)
    assert ss.lib.actions.SweepRunner.validate(sweep) is None


# %% iterate_sweep dispatch


class RecordingRunner(SweepRunner):
    """a `SweepRunner` whose `iterate` yields the resources and keyword arguments
    it received"""

    def iterate(self, resources, *, always_yield=False, yield_values=True, loop=False):
        yield (
            resources,
            {'always_yield': always_yield, 'yield_values': yield_values, 'loop': loop},
        )


def test_iterate_sweep_runs_the_runner_in_the_resources():
    """`iterate_sweep` calls the `iterate` of ``resources['runner']``. The runner
    receives the resources with the replacements applied and the modes forwarded."""
    sweep = tone_sweep(OFFSETS)
    runner = RecordingRunner(sweep)
    res = ss.lib.resources.Resources(
        action_spec=sweep, sweep_spec=sweep, runner=runner, format_path=None
    )
    modes = {'always_yield': True, 'yield_values': False, 'loop': True}

    items = list(ss.iterate_sweep(res, calibration=None, **modes))

    assert items == [({**res, 'calibration': None}, modes)]
    assert set(res) == {'action_spec', 'sweep_spec', 'runner', 'format_path'}, (
        'the input is not mutated'
    )


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
