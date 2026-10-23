"""striqt.sensor.lib.actions.registry: ActionRegistry registration and MRO lookup.

Every test builds its own `ActionRegistry` and leaves the global instance, which
the built-in runners populate at import, untouched. Spec instances come from the
bound `single_tone` class. An unbound `Sweep` subclass has no capture type and
cannot be instantiated.
"""

from __future__ import annotations

import pytest
from sweep_strategies import CalSweepCls, SweepCls, make_sweep

import striqt.sensor as ss
from striqt.sensor.lib.actions.registry import ActionRegistry


class RunnerA:
    pass


class RunnerB:
    pass


class UnboundSweep(ss.specs.Sweep, frozen=True, kw_only=True):
    """a Sweep subclass that no binding derives from"""


# %% ActionRegistry.runner


def test_runner_registers_and_returns_the_class_unchanged():
    reg = ActionRegistry()
    assert reg.runner(spec_type=ss.specs.Sweep)(RunnerA) is RunnerA
    assert reg == {ss.specs.Sweep: RunnerA}


def test_duplicate_registration_raises_naming_the_spec_type():
    reg = ActionRegistry()
    reg.runner(spec_type=UnboundSweep)(RunnerA)
    with pytest.raises(TypeError, match='UnboundSweep'):
        reg.runner(spec_type=UnboundSweep)(RunnerB)
    assert reg[UnboundSweep] is RunnerA


# %% ActionRegistry.lookup


@pytest.mark.parametrize(
    'spec',
    [SweepCls, make_sweep()],
    ids=['type', 'instance'],
)
def test_lookup_by_exact_type(spec):
    reg = ActionRegistry()
    reg.runner(spec_type=SweepCls)(RunnerA)
    assert reg.lookup(spec) is RunnerA


def test_lookup_returns_the_nearest_registered_base():
    reg = ActionRegistry()
    reg.runner(spec_type=ss.specs.Sweep)(RunnerA)
    reg.runner(spec_type=ss.specs.ManualYFactorCalibration)(RunnerB)
    assert reg.lookup(SweepCls) is RunnerA
    assert reg.lookup(CalSweepCls) is RunnerB


def test_lookup_of_an_unregistered_type_raises_naming_it():
    reg = ActionRegistry()
    reg.runner(spec_type=ss.specs.ManualYFactorCalibration)(RunnerB)
    with pytest.raises(TypeError, match='UnboundSweep'):
        reg.lookup(UnboundSweep)
