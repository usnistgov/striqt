"""striqt.analysis.lib.io: YAML decoding with `!include`, zarr store access and
allocation, and the attr repair of stores written with bandwidth-suffixed units"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr
import zarr

import striqt.analysis as sa

load = sa.lib.io.decode_from_yaml_file

ZARR_MAJOR = int(zarr.__version__.split('.')[0])

# the keys of a group or array document in either zarr format
METADATA_BASENAMES = {'.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json'}


# %% !include


def test_glob_include_merges_in_sorted_filename_order(write_yaml):
    """the files are written out of name order. A key set by 2 files takes the value
    from the file with the later name."""
    write_yaml('sites/c.yaml', 'z: 3\n')
    write_yaml('sites/a.yaml', 'x: 1\ny: 1\n')
    write_yaml('sites/b.yaml', 'y: 2\nz: 2\n')
    spec = write_yaml('spec.yaml', 'sites: !include "sites/*.yaml"\n')
    assert load(spec)['sites'] == {'x': 1, 'y': 2, 'z': 3}


@pytest.mark.parametrize(
    'order,expected',
    [
        ('[a.yaml, b.yaml]', {'x': 1, 'y': 2}),
        ('[b.yaml, a.yaml]', {'x': 1, 'y': 1}),
    ],
    ids=['a_then_b', 'b_then_a'],
)
def test_list_include_merges_dicts_with_later_files_winning(
    write_yaml, order, expected
):
    write_yaml('a.yaml', 'x: 1\ny: 1\n')
    write_yaml('b.yaml', 'y: 2\n')
    spec = write_yaml('spec.yaml', f'source: !include {order}\n')
    assert load(spec)['source'] == expected


def test_glob_include_merges_only_one_level_deep(write_yaml):
    write_yaml('sites/a.yaml', 'block:\n  x: 1\n  y: 1\n')
    write_yaml('sites/b.yaml', 'block:\n  y: 2\n')
    spec = write_yaml('spec.yaml', 'sites: !include "sites/*.yaml"\n')
    assert load(spec)['sites'] == {'block': {'y': 2}}


def test_list_include_concatenates_sequences(write_yaml):
    write_yaml('a.yaml', '- 1\n- 2\n')
    write_yaml('b.yaml', '- 3\n')
    spec = write_yaml('spec.yaml', 'items: !include [a.yaml, b.yaml]\n')
    assert load(spec)['items'] == [1, 2, 3]


def test_mixed_include_types_raise(write_yaml):
    write_yaml('a.yaml', 'x: 1\n')
    write_yaml('b.yaml', '- 3\n')
    spec = write_yaml('spec.yaml', 'items: !include [a.yaml, b.yaml]\n')
    with pytest.raises(TypeError, match='all mappings or all sequences'):
        load(spec)


def test_empty_glob_raises_with_the_pattern(write_yaml):
    spec = write_yaml('spec.yaml', 'sites: !include "missing/*.yaml"\n')
    with pytest.raises(FileNotFoundError, match=r'missing/\*\.yaml'):
        load(spec)


def test_parent_relative_glob_from_subdirectory(write_yaml):
    write_yaml('sites/global.yaml', 'defaults: {a: 1}\n')
    write_yaml('sites/radio.yaml', 'beef: {b: 2}\n')
    spec = write_yaml('site/spec.yaml', 'adjust: !include "../sites/*.yaml"\n')
    assert load(spec)['adjust'] == {'defaults': {'a': 1}, 'beef': {'b': 2}}


@pytest.mark.parametrize(
    'frag_path',
    ['frag.yaml', 'elsewhere/frag.yaml'],
    ids=['inside_root', 'outside_root'],
)
def test_absolute_include(write_yaml, frag_path):
    frag = write_yaml(frag_path, 'v: 1\n')
    spec = write_yaml('site/spec.yaml', f'top: !include {frag}\n')
    assert load(spec)['top'] == {'v': 1}


def test_nested_include_resolves_relative_to_the_including_file(write_yaml):
    write_yaml('frag/leaf.yaml', 'v: 1\n')
    write_yaml('frag/c.yaml', 'x: !include leaf.yaml\n')
    spec = write_yaml('spec.yaml', 'top: !include frag/c.yaml\n')
    assert load(spec)['top'] == {'x': {'v': 1}}


# %% loader behavior


def test_flow_sequence_mapping_keys_become_tuples(write_yaml):
    spec = write_yaml('spec.yaml', 'lookup:\n  [0, 1]: 2\n  [1, 0]: 3\n')
    assert load(spec)['lookup'] == {(0, 1): 2, (1, 0): 3}


@pytest.mark.parametrize(
    'text, expected',
    [('3750e6', '3750e6'), ('inf', 'inf'), ('.inf', math.inf)],
)
def test_scalar_typing(write_yaml, text, expected):
    # YAML 1.1 leaves engineering-notation numbers as strings, so a loop point like
    # 3750e6 reaches _build_loop_points_dict as a string and is coerced there to the
    # capture field type (xfail-audit #1)
    value = load(write_yaml('spec.yaml', f'v: {text}\n'))['v']
    assert value == expected
    assert type(value) is type(expected)


# %% repair_legacy_attrs


def legacy_store_dataset():
    """build a dataset with the bandwidth encoded in `units`, no `noise_bandwidth` on
    the bin coordinate, the sample rate as the correlator `noise_bandwidth`, and a
    store-wide `noise_bandwidth` on the channel power outputs."""
    import numpy as np
    import xarray as xr

    bins = np.array([-np.inf, -100.0, -99.0, np.inf])
    ratio_bins = np.array([-np.inf, -1.0, 0.0, 1.0, np.inf])
    ds = xr.Dataset(
        {
            'spectrogram': (
                ('capture', 'spectrogram_time'),
                np.zeros((2, 3), 'float32'),
                {
                    'standard_name': 'PSD',
                    'long_name': 'Power Spectral Density',
                    'units': 'dBm/360 kHz',
                    'noise_bandwidth': 360e3,
                },
            ),
            'spectrogram_histogram': (
                ('capture', 'spectrogram_power_bin'),
                np.zeros((2, 4), 'float32'),
                {'standard_name': 'Fraction of counts', 'noise_bandwidth': 360e3},
            ),
            'spectrogram_ratio_histogram': (
                ('capture', 'spectrogram_ratio_power_bin'),
                np.zeros((2, 5), 'float32'),
                {'standard_name': 'Fraction of counts', 'noise_bandwidth': 360e3},
            ),
            'channel_power_time_series': (
                ('capture', 'time_elapsed'),
                np.zeros((2, 3), 'float32'),
                {
                    'standard_name': 'Channel Power',
                    'units': 'dBm',
                    'noise_bandwidth': 10e6,
                    'long_name': 'Channel Power (dBm/10 MHz)',
                },
            ),
            'channel_power_histogram': (
                ('capture', 'channel_power_bin'),
                np.zeros((2, 4), 'float32'),
                {
                    'standard_name': 'Fraction of channel power readings',
                    'noise_bandwidth': 10e6,
                },
            ),
            'iq_waveform': (
                ('capture', 'iq_index'),
                np.zeros((2, 3), 'complex64'),
                {
                    'standard_name': 'IQ waveform',
                    'units': '√mW',
                    'noise_bandwidth': 10e6,
                    'long_name': 'IQ waveform (√mW/10 MHz)',
                },
            ),
            'cellular_5g_pss_correlation': (
                ('capture', 'cellular_ssb_lag'),
                np.zeros((2, 3), 'complex64'),
                {
                    'standard_name': 'PSS Cross-Covariance',
                    'units': '√mW/7.68 MHz',
                    'noise_bandwidth': 7.68e6,
                    'sample_rate': 7.68e6,
                    'subcarrier_spacing': 30e3,
                },
            ),
            'system_noise': (
                ('capture',),
                np.array([-160.0, -150.0], 'float32'),
                {'name': 'Sensor system noise spectral density', 'units': 'dBm/Hz'},
            ),
        },
        coords={
            'spectrogram_power_bin': (
                'spectrogram_power_bin',
                bins,
                {'standard_name': 'Spectrogram bin power', 'units': 'dBm/360 kHz'},
            ),
            'channel_power_bin': (
                'channel_power_bin',
                bins,
                {
                    'standard_name': 'Channel power',
                    'units': 'dBm',
                    'noise_bandwidth': 10e6,
                    'long_name': 'Channel power (dBm/10 MHz)',
                },
            ),
            'spectrogram_ratio_power_bin': (
                'spectrogram_ratio_power_bin',
                ratio_bins,
                {
                    'standard_name': 'Spectrogram cross-channel power ratio',
                    'units': 'dB/360 kHz',
                },
            ),
        },
    )
    return ds


def test_repair_legacy_attrs_rewrites_bandwidth_labels():
    ds = sa.lib.io.repair_legacy_attrs(legacy_store_dataset())

    spg = ds.spectrogram.attrs
    assert spg['units'] == 'dBm'
    assert spg['noise_bandwidth'] == pytest.approx(360e3)
    assert spg['standard_name'] == 'PSD'
    assert spg['long_name'] == 'Power Spectral Density (dBm/360 kHz)'

    coord = ds.spectrogram_power_bin.attrs
    assert coord['units'] == 'dBm'
    assert coord['noise_bandwidth'] == pytest.approx(360e3)
    assert coord['long_name'] == 'Spectrogram bin power (dBm/360 kHz)'

    ratio = ds.spectrogram_ratio_power_bin.attrs
    assert ratio['units'] == 'dB'
    assert 'noise_bandwidth' not in ratio
    assert 'noise_bandwidth' not in ds.spectrogram_ratio_histogram.attrs
    assert ds.spectrogram_histogram.attrs['noise_bandwidth'] == pytest.approx(360e3)


def test_repair_legacy_attrs_corrects_the_correlator_noise_bandwidth():
    """a stored `noise_bandwidth` equal to the sample rate is replaced by the
    subcarrier spacing, which is the noise bandwidth of the unit-energy template"""
    ds = sa.lib.io.repair_legacy_attrs(legacy_store_dataset())

    attrs = ds.cellular_5g_pss_correlation.attrs
    assert attrs['noise_bandwidth'] == pytest.approx(30e3)
    assert attrs['units'] == '√mW'
    assert attrs['long_name'] == 'PSS Cross-Covariance (√mW/30 kHz)'
    assert attrs['sample_rate'] == pytest.approx(7.68e6)


def test_repair_legacy_attrs_removes_capture_bandwidth_attrs():
    """a store-wide `noise_bandwidth` and a bandwidth-suffixed `long_name` on the
    channel power and IQ outputs are removed. The bandwidth of those outputs is set per
    capture."""
    ds = sa.lib.io.repair_legacy_attrs(legacy_store_dataset())

    names = (
        'channel_power_time_series',
        'channel_power_histogram',
        'channel_power_bin',
        'iq_waveform',
    )
    for name in names:
        assert 'noise_bandwidth' not in ds[name].attrs, name
        assert 'long_name' not in ds[name].attrs, name
    assert ds.channel_power_time_series.attrs['units'] == 'dBm'
    assert ds.channel_power_bin.attrs['units'] == 'dBm'
    assert ds.iq_waveform.attrs['units'] == '√mW'


def test_repair_legacy_attrs_keeps_spectral_densities_and_current_stores():
    """dBm/Hz is a density, not a bandwidth label. A second pass changes nothing."""
    ds = sa.lib.io.repair_legacy_attrs(legacy_store_dataset())
    assert ds.system_noise.attrs['units'] == 'dBm/Hz'
    assert ds.system_noise.attrs['standard_name'] == 'System noise'

    before = {name: dict(ds[name].attrs) for name in ds.variables}
    sa.lib.io.repair_legacy_attrs(ds)
    after = {name: dict(ds[name].attrs) for name in ds.variables}
    assert after == before


def test_load_repairs_legacy_attrs(tmp_path):
    ds = legacy_store_dataset()
    path = tmp_path / 'legacy.zarr'
    ds.to_zarr(path)

    loaded = sa.load(path)

    assert loaded.spectrogram.attrs['units'] == 'dBm'
    assert loaded.spectrogram_power_bin.attrs['noise_bandwidth'] == pytest.approx(360e3)
    assert loaded.cellular_5g_pss_correlation.attrs['noise_bandwidth'] == pytest.approx(
        30e3
    )


# %% open_store and zarr_format_kws


def store_keys(store) -> set[str]:
    if ZARR_MAJOR < 3:
        return set(store.keys())

    from zarr.core.sync import sync

    async def collect() -> set[str]:
        return {key async for key in store.list()}

    return sync(collect())


def keys_under(store, name: str) -> set[str]:
    return {key for key in store_keys(store) if key.startswith(name + '/')}


def is_metadata_key(key: str) -> bool:
    return key.rsplit('/', 1)[-1] in METADATA_BASENAMES


@pytest.mark.parametrize('mode', ['w', 'a'])
def test_open_store_zip_write_then_read(tmp_path, mode):
    path = tmp_path / 'store.zip'
    store = sa.lib.io.open_store(path, mode=mode)
    assert isinstance(store, zarr.storage.ZipStore)

    array = zarr.open_array(store=store, path='a', mode='w', shape=(3,), dtype='int64')
    array[:] = [1, 2, 3]
    store.close()

    reader = sa.lib.io.open_store(path, mode='r')
    assert isinstance(reader, zarr.storage.ZipStore)
    assert zarr.open_array(store=reader, path='a', mode='r')[:].tolist() == [1, 2, 3]
    reader.close()


def test_zarr_format_kws_follows_the_installed_major():
    if ZARR_MAJOR < 3:
        expected = {'zarr_version': 2}
    else:
        expected = {'zarr_format': 3}
    assert sa.lib.io.zarr_format_kws() == expected
    assert sa.lib.io.zarr_format_kws('auto') == expected
    assert sa.lib.io.zarr_format_kws(2) == {'zarr_version': 2}
    assert sa.lib.io.zarr_format_kws(3) == {'zarr_format': 3}


@pytest.mark.parametrize('bad', ['v2', 4, '2'])
def test_zarr_format_kws_rejects_other_arguments(bad):
    with pytest.raises(TypeError):
        sa.lib.io.zarr_format_kws(bad)


@pytest.mark.parametrize('suffix', ['.zarr', '.zip'])
def test_get_store_info_exists_only_after_a_write(tmp_path, suffix):
    store = sa.lib.io.open_store(tmp_path / f'store{suffix}', mode='w')
    exists, kws = sa.lib.io._get_store_info(store)
    assert exists is False
    assert kws == sa.lib.io.zarr_format_kws()

    zarr.open_group(store, mode='a')
    exists, _ = sa.lib.io._get_store_info(store)
    assert exists is True
    store.close()


# %% allocate_store

ROWS = 5
TEMPLATE_CHUNKS = {
    'channel_power_time_series': (2, 4),
    'iq_waveform': (2, 2),
    'start_time': (2,),
}
TIME_ENCODING = {
    'start_time': {'units': 'nanoseconds since 1970-01-01', 'dtype': 'int64'}
}
TIMES = np.array(['2026-10-30T12:00:00.000000001', '2026-10-30T12:00:01'], 'M8[ns]')


def zero_row_template() -> xr.Dataset:
    """2 registered measurements on a fixed coordinate `x` and a datetime capture
    coordinate, with 0 rows along `capture`. `channel_power_time_series` is registered
    with `store_compressed=True` and `iq_waveform` with `store_compressed=False`."""
    return xr.Dataset(
        {
            'channel_power_time_series': (
                ('capture', 'x'),
                np.zeros((0, 4), 'float32'),
            ),
            'iq_waveform': (('capture', 'x'), np.zeros((0, 4), 'complex64')),
        },
        coords={
            'x': ('x', np.arange(4.0)),
            'start_time': ('capture', np.array([], 'M8[ns]')),
        },
    )


def allocate_template(path, **kws):
    store = sa.lib.io.open_store(path, mode='w')
    arrays = sa.lib.io.allocate_store(
        store,
        zero_row_template(),
        rows=ROWS,
        chunks=TEMPLATE_CHUNKS,
        encoding=TIME_ENCODING,
        **kws,
    )
    return store, arrays


def codec_of(array):
    if ZARR_MAJOR < 3:
        return array.compressor
    elif len(array.compressors) == 0:
        return None
    else:
        return array.compressors[0]


@pytest.mark.parametrize('suffix', ['.zarr', '.zip'])
def test_allocate_store_writes_metadata_only(tmp_path, suffix):
    store, arrays = allocate_template(tmp_path / f'store{suffix}')

    assert set(arrays) == set(TEMPLATE_CHUNKS)
    for name, chunks in TEMPLATE_CHUNKS.items():
        keys = keys_under(store, name)
        assert len(keys) > 0, name
        assert all(is_metadata_key(k) for k in keys), (name, keys)
        assert tuple(arrays[name].chunks) == chunks
    assert arrays['channel_power_time_series'].shape == (ROWS, 4)
    assert arrays['iq_waveform'].shape == (ROWS, 4)
    assert arrays['start_time'].shape == (ROWS,)
    assert arrays['channel_power_time_series'].dtype == np.float32
    assert arrays['iq_waveform'].dtype == np.complex64
    assert arrays['start_time'].dtype == np.int64

    assert any(not is_metadata_key(k) for k in keys_under(store, 'x'))
    x = zarr.open_array(store=store, path='x', mode='r+')
    assert x[:].tolist() == [0.0, 1.0, 2.0, 3.0]
    store.close()


@pytest.mark.parametrize('suffix', ['.zarr', '.zip'])
def test_allocate_store_arrays_write_all_fill_chunks(tmp_path, suffix):
    """a chunk equal to the fill value is stored rather than deleted. A zip store cannot
    delete a key."""
    store, arrays = allocate_template(tmp_path / f'store{suffix}')
    array = arrays['channel_power_time_series']
    array[0:2, :] = np.full((2, 4), array.fill_value, dtype=array.dtype)
    chunk_keys = {
        k
        for k in keys_under(store, 'channel_power_time_series')
        if not is_metadata_key(k)
    }
    assert len(chunk_keys) == 1
    store.close()


def test_allocate_store_rejects_populated_templates(tmp_path):
    store = sa.lib.io.open_store(tmp_path / 'store.zarr', mode='w')
    template = zero_row_template()
    populated = xr.Dataset(
        {'channel_power_time_series': (('capture', 'x'), np.zeros((2, 4), 'float32'))},
        coords={'x': ('x', np.arange(4.0))},
    )
    with pytest.raises(ValueError, match='capture'):
        sa.lib.io.allocate_store(store, populated, rows=ROWS, chunks=TEMPLATE_CHUNKS)
    with pytest.raises(ValueError, match='capture'):
        sa.lib.io.allocate_store(
            store, template.drop_dims('capture'), rows=ROWS, chunks={}
        )


@pytest.mark.parametrize('compression', [True, False])
def test_allocate_store_codec_follows_the_registry(tmp_path, compression):
    store, arrays = allocate_template(tmp_path / 'store.zarr', compression=compression)

    codec = codec_of(arrays['channel_power_time_series'])
    if not compression:
        assert codec is None
    elif ZARR_MAJOR < 3:
        import numcodecs

        assert isinstance(codec, numcodecs.Blosc)
        assert codec.cname == 'zstd'
        assert codec.clevel == 1
    else:
        from zarr.codecs import BloscCodec

        assert isinstance(codec, BloscCodec)
        assert getattr(codec.cname, 'value', codec.cname) == 'zstd'
        assert codec.clevel == 1

    assert codec_of(arrays['iq_waveform']) is None
    store.close()


@pytest.mark.parametrize('suffix', ['.zarr', '.zip'])
def test_allocate_store_region_write_round_trips(tmp_path, suffix):
    path = tmp_path / f'store{suffix}'
    store, arrays = allocate_template(path)

    power = np.arange(8, dtype='float32').reshape(2, 4)
    iq = (np.arange(8) + 1j * np.arange(8)).astype('complex64').reshape(2, 4)
    arrays['channel_power_time_series'][0:2, :] = power
    arrays['iq_waveform'][0:2, :] = iq
    arrays['start_time'][0:2] = TIMES.astype('int64')
    store.close()

    ds = sa.load(path, consolidated=False)
    assert ds.sizes == {'capture': ROWS, 'x': 4}
    assert ds.channel_power_time_series.dtype == np.float32
    assert ds.iq_waveform.dtype == np.complex64
    assert ds.start_time.dtype == np.dtype('M8[ns]')
    np.testing.assert_array_equal(ds.channel_power_time_series.values[:2], power)
    np.testing.assert_array_equal(ds.iq_waveform.values[:2], iq)
    np.testing.assert_array_equal(ds.start_time.values[:2], TIMES)
    np.testing.assert_array_equal(ds.x.values, np.arange(4.0))
    ds.close()


def test_allocate_store_second_call_adds_a_variable(tmp_path):
    path = tmp_path / 'store.zarr'
    store, first = allocate_template(path)
    power = np.arange(8, dtype='float32').reshape(2, 4)
    first['channel_power_time_series'][0:2, :] = power

    extra = xr.Dataset(
        {'power_spectral_density': (('capture', 'y'), np.zeros((0, 3), 'float32'))},
        coords={'y': ('y', np.arange(3.0))},
    )
    second = sa.lib.io.allocate_store(
        store, extra, rows=ROWS, chunks={'power_spectral_density': (ROWS, 3)}
    )
    assert set(second) == {'power_spectral_density'}
    assert second['power_spectral_density'].shape == (ROWS, 3)
    assert tuple(second['power_spectral_density'].chunks) == (ROWS, 3)
    assert all(is_metadata_key(k) for k in keys_under(store, 'power_spectral_density'))

    reopened = zarr.open_array(store=store, path='channel_power_time_series', mode='r+')
    assert reopened.shape == (ROWS, 4)
    np.testing.assert_array_equal(reopened[0:2, :], power)
    assert zarr.open_array(store=store, path='iq_waveform', mode='r+').shape == (
        ROWS,
        4,
    )
    store.close()

    ds = sa.load(path, consolidated=False)
    assert set(ds.data_vars) == {
        'channel_power_time_series',
        'iq_waveform',
        'power_spectral_density',
    }
    assert ds.sizes == {'capture': ROWS, 'x': 4, 'y': 3}
    ds.close()
