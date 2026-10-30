"""striqt.sensor.lib.compute.datasets: packaging delayed results into xarray datasets
and the looped-coordinate indexing helpers
"""

from __future__ import annotations

import dataclasses
import functools
import logging

import numpy as np
import pytest
from numeric_checks import assert_close, elementwise_rtol
from soapy_factories import soapy_capture
from synthetic_sources import (
    IQ_ONLY,
    SCALE_ONLY,
    SPECTROGRAM,
    preset_capture,
    preset_sweep,
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
        preset_capture('single_tone', **fields, frequency_offset=f, **kws)
        for f in OFFSETS
    )


def delayed_results(sweep) -> list[compute.DelayedDataset]:
    with ss.open_resources(sweep, None) as res:
        sink = ss.sinks.NoSink(sweep)
        return [dd for dd in ss.iterate_sweep(res, sink=sink) if dd is not None]


@functools.lru_cache
def two_port_run() -> list[compute.DelayedDataset]:
    """return the delayed results of 2 captures (`frequency_offset`) x 2 ports, run once
    per module"""
    return delayed_results(
        preset_sweep('single_tone', two_port_captures(), analysis=IQ_ONLY, loops=LOOPS)
    )


@functools.lru_cache
def spectrogram_run() -> list:
    sweep = preset_sweep('single_tone', two_port_captures(), analysis=SPECTROGRAM)
    return [compute.from_delayed(dd) for dd in delayed_results(sweep)]


# %% from_delayed


def test_from_delayed_does_not_warn_on_an_adc_overload(caplog):
    """`from_delayed` logs nothing for a negative headroom. The overload warning belongs
    to the `log_overloads` callback."""
    air = ss.bindings.air7101b
    capture = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)
    sweep = air.sensor.sweep_spec_cls(source=air.schema.source(), captures=(capture,))
    config = compute.EvaluationOptions(
        sweep_spec=sweep, registry=sa.registry, as_xarray='delayed', extra_attrs={}
    )
    dd = compute.DelayedDataset(
        delayed={},
        capture=capture,
        extra_coords=ss.specs.AcquisitionInfo(),
        extra_data={'adc_headroom': (-1.0, -1.0)},
        config=config,
    )
    with caplog.at_level(logging.WARNING, logger='striqt.analysis'):
        ds = compute.from_delayed(dd)
    assert ds.sizes['capture'] == 2
    assert caplog.records == []


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


def test_from_delayed_leaves_the_signal_trigger_out_of_the_coordinates():
    """the `signal_trigger` field of the acquisition info is a `Trigger` object and not
    a coordinate. The other info fields are coordinates."""
    trigger = sa.Trigger(
        'cellular_5g_pss_sync',
        sa.specs.Cellular5GNPSSSync(subcarrier_spacing=30e3),
        sa.registry,
    )
    info = ss.specs.AcquisitionInfo(signal_trigger=trigger, sweep_index=3)
    ds = compute.from_delayed(dataclasses.replace(two_port_run()[0], extra_coords=info))
    assert 'signal_trigger' not in ds.coords
    assert list(ds.sweep_index.values) == [3, 3]


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
    """current behavior, not a contract: a Repeat has no field and is listed as None"""
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


# %% drop_conflicts / measurement_grids / sweep_template


def test_drop_conflicts_keeps_agreeing_keys_and_drops_differing_ones():
    """the ``combine_attrs='drop_conflicts'`` rule of `xarray.concat`: a key in one
    mapping is kept, equal values are kept, differing values are dropped for good"""
    merged = datasets.drop_conflicts([
        {'a': 1, 'b': [1, 2], 'c': 'x'},
        {'a': 1, 'b': [1, 2], 'd': 4},
        {'c': 'y', 'a': 1},
    ])
    assert merged == {'a': 1, 'b': [1, 2], 'd': 4}


def mixed_rate_sweep():
    """2 single-tone captures at 2 sample rates and bandwidths, so that the frequency
    grids of the PSD differ and the time axis of the IQ differs"""
    captures = (
        preset_capture('single_tone', sample_rate=6.144e6),
        preset_capture('single_tone', sample_rate=3.072e6, analysis_bandwidth=2.5e6),
    )
    return preset_sweep('single_tone', captures)


@functools.lru_cache
def mixed_rate_reference():
    """the outer-join concatenation that the production sink writes for `mixed_rate_sweep`"""
    import xarray as xr

    sweep = mixed_rate_sweep()
    results = [compute.from_delayed(dd) for dd in delayed_results(sweep)]
    return xr.concat(results, 'capture', join='outer', combine_attrs='drop_conflicts')


def test_sweep_template_matches_the_outer_join_geometry(subtests):
    """without IQ, the template has the union grids, coordinate values and attrs, and
    the spec attrs of `xarray.concat(join='outer', combine_attrs='drop_conflicts')`"""
    sweep = mixed_rate_sweep()
    ref = mixed_rate_reference()
    delayed = delayed_results(sweep)
    template, grids, encoding, encoders = datasets.sweep_template(
        sweep, 'x', ss.specs.AcquisitionInfo, attrs={}
    )

    assert set(template.data_vars) == set(ref.data_vars)
    assert grids['power_spectral_density'].mixed
    assert not grids['channel_power_time_series'].mixed
    for name in ref.data_vars:
        with subtests.test(variable=name):
            assert template[name].dims == ref[name].dims
            assert template[name].shape[1:] == ref[name].shape[1:]
            assert template[name].dtype == ref[name].dtype
            for dim in ref[name].dims[1:]:
                assert np.array_equal(template[dim].values, ref[dim].values)
                assert dict(template[dim].attrs) == dict(ref[dim].attrs)
            # the attrs that a measurement body returns are known only with data
            body_keys = {k for dd in delayed for k in dd.delayed[name].attrs}
            expected = {k: v for k, v in ref[name].attrs.items() if k not in body_keys}
            got = {k: v for k, v in template[name].attrs.items() if k not in body_keys}
            assert got == expected
    assert template.sizes['capture'] == 0
    assert set(encoders) == {
        str(k) for k, v in ref.coords.items() if 'capture' in v.dims
    }
    assert encoding == {}


def test_sweep_template_capture_coordinates_have_storage_dtypes():
    """string coordinates become fixed-width unicode wide enough for every capture and
    at least 16 characters; numeric and bool coordinates keep their dtype"""
    sweep = mixed_rate_sweep()
    template, *_ = datasets.sweep_template(
        sweep, 'a_source_id_longer_than_16_chars', ss.specs.AcquisitionInfo, attrs={}
    )
    assert template['lo_shift'].dtype == np.dtype('<U16')
    assert template['source_id'].dtype == np.dtype('<U32')
    assert template['host_resample'].dtype == np.dtype(bool)
    assert template['sample_rate'].dtype == np.dtype('float64')


def test_sweep_template_attrs_hold_the_shared_capture_fields_and_the_given_attrs():
    """the dataset attrs are the capture fields that are equal on every capture and
    are not coordinates, as the production concat leaves them, plus the given attrs"""
    sweep = mixed_rate_sweep()
    ref = mixed_rate_reference()
    sweep_level = set(sweep.to_dict(True, allow_tuple_keys=False))
    capture_level = {k: v for k, v in ref.attrs.items() if k not in sweep_level}

    template, *_ = datasets.sweep_template(
        sweep, 'x', ss.specs.AcquisitionInfo, attrs={'sensor_binding': 'single_tone'}
    )
    assert dict(template.attrs) == {**capture_level, 'sensor_binding': 'single_tone'}
    assert 'sample_rate' not in template.attrs


def test_row_index_and_scatter_rows_reproduce_the_outer_join_fill():
    """a capture's rows land where `xarray.align(join='outer')` puts them, with NaN
    where the capture has no value"""
    sweep = mixed_rate_sweep()
    ref = mixed_rate_reference()
    _, grids, *_ = datasets.sweep_template(
        sweep, 'x', ss.specs.AcquisitionInfo, attrs={}
    )
    grid = grids['power_spectral_density']
    delayed = delayed_results(sweep)

    for dd in delayed:
        spec = dd.config.sweep_spec.analysis.power_spectral_density
        index = datasets.row_index(grid, dd.capture, spec)
        block = datasets.scatter_rows(
            grid, dd.delayed['power_spectral_density'].result, index
        )
        rows = ref.power_spectral_density.sel(
            capture=ref.sample_rate == dd.capture.sample_rate
        )
        assert block.shape == rows.shape
        np.testing.assert_array_equal(block, rows.values)


def test_scatter_rows_returns_the_input_on_an_identical_grid():
    sweep = mixed_rate_sweep()
    _, grids, *_ = datasets.sweep_template(
        sweep, 'x', ss.specs.AcquisitionInfo, attrs={}
    )
    grid = grids['channel_power_time_series']
    rows = np.zeros((2, *grid.shape), dtype=grid.dtype)
    index = tuple(slice(None) for _ in grid.dims)
    assert datasets.scatter_rows(grid, rows, index) is rows


def test_row_index_rejects_a_value_off_the_grid():
    sweep = mixed_rate_sweep()
    _, grids, *_ = datasets.sweep_template(
        sweep, 'x', ss.specs.AcquisitionInfo, attrs={}
    )
    grid = grids['power_spectral_density']
    off_grid = dataclasses.replace(
        grid,
        coords={
            **grid.coords,
            'baseband_frequency': grid.coords['baseband_frequency'][1:],
        },
    )
    capture = sweep.captures[0]
    with pytest.raises(ValueError):
        datasets.row_index(off_grid, capture, sweep.analysis.power_spectral_density)


# %% capture_coord_storage


@pytest.mark.parametrize(
    'values, known, dtype, encoded',
    [
        (np.array(['none', 'left'], dtype=object), (), '<U16', ['none', 'left']),
        (np.array(['ab'], dtype=object), ('x' * 20,), '<U20', ['ab']),
        (np.array([1.5, 2.5]), (), 'float64', [1.5, 2.5]),
        (np.array([True, False]), (), 'bool', [True, False]),
    ],
    ids=['str', 'str_widened_by_known', 'float', 'bool'],
)
def test_capture_coord_storage_plain_dtypes(values, known, dtype, encoded):
    storage, encoding, encoder = datasets.capture_coord_storage(values, known)
    assert storage == np.dtype(dtype)
    assert encoding == {}
    assert encoder(values).tolist() == encoded


def test_capture_coord_storage_encodes_timestamps_as_int64_nanoseconds():
    import pandas as pd

    values = np.array([pd.Timestamp('2026-01-01T00:00:00.000000001')], dtype=object)
    storage, encoding, encoder = datasets.capture_coord_storage(values)
    assert storage == np.dtype('datetime64[ns]')
    assert encoding == datasets.TIME_ENCODING
    seconds = (pd.Timestamp('2026-01-01') - pd.Timestamp(0)) // pd.Timedelta(1, 's')
    assert encoder(values).tolist() == [seconds * 10**9 + 1]
    assert encoder(np.array(['NaT'], dtype='datetime64[ns]')).tolist() == [
        datasets.TIME_ENCODING['_FillValue']
    ]


# %% extra_data_arrays


def test_extra_data_arrays_names_the_dims_and_broadcasts_scalars():
    arrays = datasets.extra_data_arrays({'t': 1.5, 'v': np.arange(6).reshape(2, 3)}, 2)
    assert arrays['t'].dims == ('capture',)
    assert arrays['t'].values.tolist() == [1.5, 1.5]
    assert arrays['v'].dims == ('capture', 'v_dim1')
    assert arrays['v'].shape == (2, 3)
