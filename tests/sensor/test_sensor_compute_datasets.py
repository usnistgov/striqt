"""striqt.sensor.lib.compute.datasets: packaging a DelayedDataset into an
xarray.Dataset (capture rows, per-port coordinates, attrs, peripheral data), the
spectrogram time concatenation, and the looped-coordinate indexing helpers"""

from __future__ import annotations

import dataclasses
import functools
import logging

import numpy as np
import pytest
from numeric_checks import assert_close, elementwise_rtol
from synthetic_sources import (
    IQ_ONLY,
    SCALE_ONLY,
    SPECTROGRAM,
    make_capture,
    make_sweep,
    spectrogram_frames,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute
from striqt.sensor.lib.compute import datasets

OFFSETS = (1e6, 2e6)
LO_FREQUENCIES = (10e9, 10.5e9)
LOOPS = (ss.specs.Repeat(count=1), ss.specs.List(field='snr', values=(None,)))
RESOLUTION_LOOP = ss.specs.List(
    field='frequency_resolution', isin='analysis', values=(8e3, 16e3)
)
WINDOW_LOOP = ss.specs.List(field='window', isin='analysis', values=('hann', 'hamming'))


def two_port_captures(**kws):
    fields = {**SCALE_ONLY, 'snr': None, 'external_lo_frequency': LO_FREQUENCIES}
    return tuple(
        make_capture('single_tone', **fields, frequency_offset=f, **kws)
        for f in OFFSETS
    )


def delayed_results(sweep) -> list[compute.DelayedDataset]:
    with ss.open_resources(sweep, None) as res:
        sink = ss.sinks.NoSink(sweep)
        return [dd for dd in ss.iterate_sweep(res, sink=sink) if dd is not None]


@functools.lru_cache
def two_port_run() -> list[compute.DelayedDataset]:
    """2 captures (frequency_offset) x 2 ports; run once for the module"""
    return delayed_results(
        make_sweep('single_tone', two_port_captures(), analysis=IQ_ONLY, loops=LOOPS)
    )


@functools.lru_cache
def spectrogram_run() -> list:
    sweep = make_sweep('single_tone', two_port_captures(), analysis=SPECTROGRAM)
    return [compute.from_delayed(dd) for dd in delayed_results(sweep)]


def options(**kws):
    sweep = make_sweep('single_tone', two_port_captures(), analysis=IQ_ONLY)
    return compute.EvaluationOptions(
        sweep_spec=sweep, registry=sa.registry, as_xarray='delayed', **kws
    )


# %% EvaluationOptions


def test_evaluation_options_default_extra_attrs_is_an_empty_dict():
    assert options().extra_attrs == {}


# %% from_delayed


def test_from_delayed_one_row_per_port_with_split_fields():
    ds = compute.from_delayed(two_port_run()[1])
    assert ds.sizes['capture'] == 2
    assert ds.port.values.tolist() == [0, 1]
    assert ds.external_lo_frequency.values.tolist() == list(LO_FREQUENCIES)
    assert ds.frequency_offset.values.tolist() == [OFFSETS[1]] * 2
    assert ds.capture_index.values.tolist() == [1, 1]
    assert ds.sweep_index.values.tolist() == [0, 0]


def test_from_delayed_attrs_hold_the_sweep_but_not_the_coordinates():
    ds = compute.from_delayed(two_port_run()[0])
    assert ds.attrs['sensor_binding'] == 'single_tone'
    assert ds.attrs['loops'] == tuple(loop.to_dict() for loop in LOOPS)
    assert set(ds.attrs).isdisjoint(ds.coords)


def test_from_delayed_broadcasts_extra_data_over_the_rows():
    dd = dataclasses.replace(
        two_port_run()[0],
        extra_data={'temperature': 21.5, 'per_port': np.array([1.0, 2.0])},
    )
    ds = compute.from_delayed(dd)
    assert ds.temperature.dims == ('capture',)
    assert ds.temperature.values.tolist() == [21.5, 21.5]
    assert ds.per_port.values.tolist() == [1.0, 2.0]


def test_from_delayed_rejects_extra_data_of_the_wrong_length():
    dd = dataclasses.replace(two_port_run()[0], extra_data={'bad': np.zeros(3)})
    with pytest.raises(ValueError, match='"bad"'):
        compute.from_delayed(dd)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason='_coords_template types sweep_index as int from Union[int, None], so '
    'build_capture_coords cannot store the AcquisitionInfo default None '
    '(datasets.py:194)',
)
def test_from_delayed_accepts_the_acquisition_info_defaults():
    dd = dataclasses.replace(two_port_run()[0], extra_coords=ss.specs.AcquisitionInfo())
    compute.from_delayed(dd)


# %% build_capture_coords


def test_build_capture_coords_splits_tuple_fields_per_port():
    capture = two_port_captures()[0]
    info = ss.specs.AcquisitionInfo(source_id='beef', sweep_index=3, capture_index=7)
    coords = datasets.build_capture_coords(capture, info, ())
    assert coords['port'].values.tolist() == [0, 1]
    assert coords['external_lo_frequency'].values.tolist() == list(LO_FREQUENCIES)
    assert coords['frequency_offset'].values.tolist() == [OFFSETS[0]] * 2
    assert coords['source_id'].values.tolist() == ['beef'] * 2
    assert coords['sweep_index'].values.tolist() == [3, 3]
    assert coords['capture_index'].values.tolist() == [7, 7]


def test_build_capture_coords_adds_an_analysis_loop_coordinate():
    capture = two_port_captures(adjust_analysis={'frequency_resolution': 16e3})[0]
    coords = datasets.build_capture_coords(
        capture, ss.specs.AcquisitionInfo(sweep_index=0), (RESOLUTION_LOOP,)
    )
    assert coords['frequency_resolution'].values.tolist() == [16e3, 16e3]


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason='AnalysisRegistry stores parameter_fields[name] = None when '
    'infer_coord_info rejects the field type (analysis/lib/register.py:401), and '
    '_coords_template then calls make_var(None) for an analysis loop over it '
    '(datasets.py:327)',
)
def test_build_capture_coords_adds_a_window_loop_coordinate():
    capture = two_port_captures(adjust_analysis={'window': 'hamming'})[0]
    coords = datasets.build_capture_coords(
        capture, ss.specs.AcquisitionInfo(sweep_index=0), (WINDOW_LOOP,)
    )
    assert coords['window'].values.tolist() == ['hamming', 'hamming']


def test_build_capture_coords_rejects_an_unregistered_analysis_loop():
    loop = ss.specs.List(field='not_a_parameter', isin='analysis', values=(1,))
    with pytest.raises(KeyError, match='not_a_parameter'):
        datasets.build_capture_coords(
            two_port_captures()[0], ss.specs.AcquisitionInfo(sweep_index=0), (loop,)
        )


# %% concat_time_dim


def test_concat_time_dim_places_the_frames_back_to_back():
    first, second = spectrogram_run()
    frames, hop_period = spectrogram_frames(
        two_port_captures()[0], SPECTROGRAM.spectrogram
    )
    assert first.sizes['spectrogram_time'] == frames

    joined = compute.concat_time_dim([first, second], 'spectrogram_time')

    assert joined.sizes['spectrogram_time'] == 2 * frames
    assert joined.sizes['capture'] == first.sizes['capture']
    time = joined.spectrogram_time.values.astype('float64')
    assert_close(np.diff(time), hop_period, rtol=elementwise_rtol(np.float32))
    assert np.array_equal(
        joined.spectrogram.values[0, :frames], first.spectrogram.values[0]
    )
    assert np.array_equal(
        joined.spectrogram.values[0, frames:], second.spectrogram.values[0]
    )


# %% get_looped_coords


def test_get_looped_coords_lists_the_capture_loop_fields():
    ds = compute.from_delayed(two_port_run()[0])
    assert compute.get_looped_coords(ds) == ['snr']


def test_get_looped_coords_include_repeats():
    """current behaviour, not a contract: a Repeat has no field and is listed as None"""
    ds = compute.from_delayed(two_port_run()[0])
    assert compute.get_looped_coords(ds, include_repeats=True) == [None, 'snr']


def test_get_looped_coords_requires_the_loops_attr():
    ds = compute.from_delayed(two_port_run()[0])
    del ds.attrs['loops']
    with pytest.raises(AttributeError, match='loops'):
        compute.get_looped_coords(ds)


# %% index_dataset / unstack_dataset


def full_run():
    import xarray as xr

    return xr.concat([compute.from_delayed(dd) for dd in two_port_run()], 'capture')


def test_unstack_dataset_forms_the_2x2_grid():
    ds = full_run()
    unstacked = compute.unstack_dataset(ds, ['frequency_offset', 'port'])
    assert unstacked.sizes['frequency_offset'] == 2
    assert unstacked.sizes['port'] == 2
    assert unstacked.frequency_offset.values.tolist() == list(OFFSETS)
    assert unstacked.port.values.tolist() == [0, 1]
    # row 3 of the flat dataset is (capture 1, port 1)
    cell = unstacked.iq_waveform.sel(frequency_offset=OFFSETS[1], port=1)
    assert np.array_equal(cell.values, ds.iq_waveform.values[3])


def test_index_dataset_selects_by_the_coordinate_pair():
    ds = full_run()
    indexed = compute.index_dataset(ds, ['frequency_offset', 'port'])
    row = indexed.iq_waveform.sel(frequency_offset=OFFSETS[0], port=1)
    assert np.array_equal(row.values, ds.iq_waveform.values[1])


def test_check_coord_indexes_rejects_a_non_unique_index():
    if datasets._xarray_version() < (2024, 9, 0):
        pytest.skip('xarray is too old to validate coord indexes')
    with pytest.raises(ValueError, match='insufficient'):
        compute.index_dataset(full_run(), ['port'])


def test_check_coord_indexes_warns_on_old_xarray(monkeypatch, caplog):
    monkeypatch.setattr(datasets, '_xarray_version', lambda: (2024, 8, 0))
    with caplog.at_level(logging.WARNING, logger='striqt.analysis'):
        datasets._check_coord_indexes(full_run(), ['port'])
    assert [r.getMessage() for r in caplog.records] == [
        'xarray is too old to to validate coord indexes'
    ]
