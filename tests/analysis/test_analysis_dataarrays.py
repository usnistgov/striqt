"""striqt.analysis.lib.dataarrays: the result containers that `EvaluationOptions`
selects when an analysis group is evaluated on a waveform.

The measurements are chosen for cost: a 1000-sample tone through the IQ passthrough
and the channel power detectors, which run without a window design or a resampler.
`Preprocessor` is checked on the base class, which returns `x` for every preference,
and through a recording subclass that records the preferences the registry passes.
The power-units cell runs a 10 ms tone at 1.92 MS/s through one group with an output
in each form of a power unit: dB, amplitude, linear power, and a power-bin coordinate,
once per power unit in a module-scoped fixture.
"""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest
import xarray as xr
from analysis_strategies import POWER_BINS, RecordingIQ

import striqt.analysis as sa
import striqt.waveform as sw
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
    """return the array that the measurement registered as `name` returns for `iq`
    with ``as_xarray=False``."""
    spec = getattr(GROUP, name)
    data, _ = sa.registry[type(spec)].func(
        iq, CAPTURE, as_xarray=False, **spec.to_dict()
    )
    return data


# %% Preprocessor


def test_the_base_preprocessor_returns_x_for_every_preference() -> None:
    x = np.zeros((2, 8), dtype='complex64')
    iq = sa.AcquiredIQ(x=x, capture=None, power_units='a.u.')
    for prefer_filtered in (True, False):
        for prefer_aligned in (True, False):
            out = iq.preprocess(
                prefer_filtered=prefer_filtered, prefer_aligned=prefer_aligned
            )
            assert out is x, (prefer_filtered, prefer_aligned)


def test_a_preprocessor_outliving_its_acquisition_raises() -> None:
    iq = sa.AcquiredIQ(
        x=np.zeros((2, 8), dtype='complex64'), capture=None, power_units='a.u.'
    )
    preprocess = iq.preprocess
    del iq
    gc.collect()
    with pytest.raises(RuntimeError):
        preprocess()


# %% evaluate_by_spec


def test_raw_and_delayed_results_are_the_registered_arrays(iq: np.ndarray) -> None:
    """``as_xarray=False`` returns the raw array of each measurement of the group in
    group order, ``'delayed'`` returns a `DelayedDataArray` with that array as the
    result, and `analyze_by_spec` dispatches to the same results"""
    raw = sa.dataarrays.evaluate_by_spec(iq, GROUP, CAPTURE, options(as_xarray=False))
    delayed = sa.dataarrays.evaluate_by_spec(
        iq, GROUP, CAPTURE, options(as_xarray='delayed')
    )
    dispatched = sa.analyze_by_spec(iq, GROUP, CAPTURE, options(as_xarray=False))

    assert isinstance(raw, dict)
    assert list(raw) == list(delayed) == list(dispatched) == list(GROUP.to_dict())
    for name in raw:
        expected = raw_measurement(iq, name)
        assert isinstance(raw[name], np.ndarray)
        assert np.array_equal(raw[name], expected)
        assert isinstance(delayed[name], DelayedDataArray)
        assert np.array_equal(delayed[name].result, expected)
        assert np.array_equal(dispatched[name], expected)


def test_evaluate_by_spec_hands_the_acquisition_to_the_measurement() -> None:
    """a measurement given an `AcquiredIQ` receives the array that the `preprocess` of
    the acquisition returns for the preferences registered with the measurement"""
    capture = sa.specs.Capture(duration=16e-6, sample_rate=1e6)
    x = np.arange(32, dtype='complex64').reshape(2, 16)
    iq = RecordingIQ(x=x, capture=capture, power_units='a.u.')
    spec = sa.specs.BuiltinAnalyses(iq_waveform=sa.specs.IQWaveform())
    options = sa.EvaluationOptions(as_xarray='delayed', registry=sa.registry)

    results = sa.analyze_by_spec(iq, spec, capture, options)

    assert np.array_equal(results['iq_waveform'].result, x + 1)
    assert iq.preprocess.calls == [(True, True)]


# %% analyze_by_spec


def test_default_options_build_a_dataset_over_port(iq: np.ndarray) -> None:
    ds = sa.analyze_by_spec(iq, GROUP, CAPTURE)
    explicit = sa.analyze_by_spec(iq, GROUP, CAPTURE, options(as_xarray=True))

    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == set(GROUP.to_dict())
    assert ds.sizes['port'] == iq.shape[0]
    assert ds.iq_waveform.dims[0] == 'port'
    assert ds.identical(explicit)


# %% power units

POWER_UNITS_CAPTURE = sa.specs.Capture(duration=10e-3, sample_rate=1.92e6)
_STFT = {'window': 'hamming', 'frequency_resolution': 30e3}
POWER_UNITS_GROUP = sa.specs.BuiltinAnalyses.from_dict({
    'channel_power_time_series': {'detector_period': 1e-3},
    'iq_waveform': {},
    'spectrogram': _STFT,
    'spectrogram_histogram': {**_STFT, **POWER_BINS},
    'spectrogram_ratio_histogram': {**_STFT, **POWER_BINS},
    'cellular_cyclic_autocorrelation': {'subcarrier_spacings': 15e3},
})

# the units attrs of the outputs and power-bin coordinates of POWER_UNITS_GROUP for
# each power unit of the acquisition. The registry declares the mW row.
POWER_UNITS_EXPECTED = {
    'mW': {
        'channel_power_time_series': 'dBm',
        'spectrogram': 'dBm',
        'iq_waveform': '√mW',
        'cellular_cyclic_autocorrelation': 'mW',
        'spectrogram_power_bin': 'dBm',
        'spectrogram_ratio_power_bin': 'dB',
    },
    'FS': {
        'channel_power_time_series': 'dBFS',
        'spectrogram': 'dBFS',
        'iq_waveform': '√FS',
        'cellular_cyclic_autocorrelation': 'FS',
        'spectrogram_power_bin': 'dBFS',
        'spectrogram_ratio_power_bin': 'dB',
    },
    'a.u.': {
        'channel_power_time_series': 'dB(a.u.)',
        'spectrogram': 'dB(a.u.)',
        'iq_waveform': '√a.u.',
        'cellular_cyclic_autocorrelation': 'a.u.',
        'spectrogram_power_bin': 'dB(a.u.)',
        'spectrogram_ratio_power_bin': 'dB',
    },
}


@pytest.fixture(scope='module')
def power_units_x() -> np.ndarray:
    return sa.testing.tone(
        POWER_UNITS_CAPTURE.duration, POWER_UNITS_CAPTURE.sample_rate, ports=2
    )


def units_of(iq: np.ndarray | sa.AcquiredIQ) -> dict[str, str]:
    """return the `units` attr of each variable and power-bin coordinate of
    `POWER_UNITS_GROUP` evaluated on `iq`, keyed by name."""
    ds = sa.analyze_by_spec(iq, POWER_UNITS_GROUP, POWER_UNITS_CAPTURE)
    return {
        str(name): ds[name].attrs['units']
        for name in ds.variables
        if name in ds.data_vars or str(name).endswith('_power_bin')
        if 'units' in ds[name].attrs
    }


@pytest.fixture(scope='module')
def units_by_power_units(power_units_x: np.ndarray) -> dict[str, dict[str, str]]:
    """the `units_of` result for an acquisition in each power unit, evaluated in the
    order FS, a.u., mW, and then for the bare array under ``'bare'``"""
    results = {}
    for power_units in ('FS', 'a.u.', 'mW'):
        iq = sa.AcquiredIQ(power_units_x, POWER_UNITS_CAPTURE, power_units=power_units)
        results[power_units] = units_of(iq)
    results['bare'] = units_of(power_units_x)
    return results


@pytest.mark.parametrize('power_units', list(POWER_UNITS_EXPECTED))
def test_power_units_of_the_acquisition_label_every_power_output(
    units_by_power_units: dict[str, dict[str, str]], power_units: sw.units.PowerUnits
) -> None:
    """the dB, amplitude, and linear power outputs and the power-bin coordinate take
    the family of `AcquiredIQ.power_units`; the ratio coordinate stays in dB"""
    assert units_by_power_units[power_units] == POWER_UNITS_EXPECTED[power_units]


def test_a_bare_array_is_in_arbitrary_units(
    units_by_power_units: dict[str, dict[str, str]],
) -> None:
    assert units_by_power_units['bare'] == POWER_UNITS_EXPECTED['a.u.']


def test_power_units_leave_the_registered_and_cached_attrs_in_mW(
    units_by_power_units: dict[str, dict[str, str]],
) -> None:
    """an evaluation in full scale resolves copies: the registered attrs and the cached
    coordinate factory result keep the form token after the evaluations of the
    fixture"""
    info = sa.registry[sa.specs.ChannelPowerTimeSeries]
    assert info.attrs['units'] == sa.units.DB_POWER
    info = sa.registry[sa.specs.SpectrogramHistogram]
    spec = POWER_UNITS_GROUP.spectrogram_histogram
    assert spec is not None
    coords = sa.dataarrays.measurement_coords(info, POWER_UNITS_CAPTURE, spec)
    assert coords['spectrogram_power_bin'][1]['units'] == sa.units.DB_POWER


def test_to_xarray_leaves_the_delayed_attrs_as_the_form(
    power_units_x: np.ndarray,
) -> None:
    """the attrs that a measurement body returns keep the form token after the
    DataArray is built in a family"""
    iq = sa.AcquiredIQ(power_units_x, POWER_UNITS_CAPTURE, power_units='FS')
    results = sa.analyze_by_spec(
        iq, POWER_UNITS_GROUP, POWER_UNITS_CAPTURE, options(as_xarray='delayed')
    )
    delayed = results['spectrogram']

    da = delayed.to_xarray(expand_dims=('port',))

    assert da.attrs['units'] == 'dBFS'
    assert delayed.attrs['units'] == sa.units.DB_POWER


def test_compute_keeps_the_power_units(power_units_x: np.ndarray) -> None:
    iq = sa.AcquiredIQ(power_units_x, POWER_UNITS_CAPTURE, power_units='FS')
    results = sa.dataarrays.evaluate_by_spec(
        iq, GROUP, POWER_UNITS_CAPTURE, options(as_xarray='delayed', block_each=False)
    )
    delayed = results['iq_waveform']
    assert isinstance(delayed, DelayedDataArray)

    assert delayed.power_units == 'FS'
    assert delayed.compute().power_units == 'FS'


# %% measurement_attrs


def test_measurement_attrs_layers_and_resolves_the_form():
    """the registered attrs, the spec fields, and the body attrs merge in that order,
    and a power form in `units` resolves into the family"""
    info = sa.registry[sa.specs.ChannelPowerTimeSeries]
    spec = sa.specs.ChannelPowerTimeSeries(detector_period=1e-3)
    attrs = sa.dataarrays.measurement_attrs(
        info, spec, {'units': sa.units.AMPLITUDE, 'extra': 1}, 'FS'
    )
    assert attrs['units'] == '√FS'
    assert attrs['detector_period'] == 1e-3
    assert attrs['extra'] == 1
    assert attrs['standard_name'] == info.attrs['standard_name']
    assert info.attrs['units'] == sa.units.DB_POWER, 'the registration is untouched'
