"""striqt.analysis.lib.dataarrays: the result containers that `EvaluationOptions`
selects when an analysis group is evaluated on a waveform.

The measurements are chosen for cost: a 1000-sample tone through the IQ passthrough
and the channel power detectors, which run without a window design or a resampler.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

import striqt.analysis as sa
from striqt.analysis.lib.dataarrays import DelayedDataArray

CAPTURE = sa.specs.Capture(duration=1e-3, sample_rate=1e6)
GROUP = sa.specs.BuiltinAnalyses.from_dict({
    'channel_power_time_series': {'detector_period': 1e-4},
    'iq_waveform': {},
})


def options(**kws: Any) -> sa.EvaluationOptions:
    return sa.EvaluationOptions(registry=sa.registry, **kws)


@pytest.fixture(scope='module')
def iq() -> np.ndarray:
    return sa.testing.tone(CAPTURE.duration, CAPTURE.sample_rate, ports=2)


def raw_measurement(iq: np.ndarray, name: str) -> np.ndarray:
    """the array that the registered measurement `name` returns on `iq` when called
    with ``as_xarray=False``"""
    spec = getattr(GROUP, name)
    data, _ = sa.registry[type(spec)].func(
        iq, CAPTURE, as_xarray=False, **spec.to_dict()
    )
    return data


# %% evaluate_by_spec


def test_as_xarray_false_returns_the_raw_arrays(iq: np.ndarray) -> None:
    results = sa.dataarrays.evaluate_by_spec(
        iq, GROUP, CAPTURE, options(as_xarray=False)
    )

    assert isinstance(results, dict)
    assert list(results) == list(GROUP.to_dict())
    for name, data in results.items():
        assert isinstance(data, np.ndarray)
        assert np.array_equal(data, raw_measurement(iq, name))


@pytest.mark.parametrize('block_each', [True, False])
def test_as_xarray_delayed_returns_delayed_arrays(
    iq: np.ndarray, block_each: bool
) -> None:
    results = sa.dataarrays.evaluate_by_spec(
        iq, GROUP, CAPTURE, options(as_xarray='delayed', block_each=block_each)
    )

    assert list(results) == list(GROUP.to_dict())
    for name, delayed in results.items():
        assert isinstance(delayed, DelayedDataArray)
        assert np.array_equal(delayed.result, raw_measurement(iq, name))


# %% analyze_by_spec


def test_as_xarray_false_matches_evaluate_by_spec(iq: np.ndarray) -> None:
    results = sa.analyze_by_spec(iq, GROUP, CAPTURE, options(as_xarray=False))
    expected = sa.dataarrays.evaluate_by_spec(
        iq, GROUP, CAPTURE, options(as_xarray=False)
    )

    assert list(results) == list(expected)
    for name in results:
        assert np.array_equal(results[name], expected[name])


def test_default_options_build_a_dataset_over_port(iq: np.ndarray) -> None:
    ds = sa.analyze_by_spec(iq, GROUP, CAPTURE)
    explicit = sa.analyze_by_spec(iq, GROUP, CAPTURE, options(as_xarray=True))

    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == set(GROUP.to_dict())
    assert ds.sizes['port'] == iq.shape[0]
    assert ds.iq_waveform.dims[0] == 'port'
    assert ds.identical(explicit)
