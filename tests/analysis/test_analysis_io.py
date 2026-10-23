"""striqt.analysis.lib.io: YAML decoding with `!include`"""

from __future__ import annotations

import math

import pytest

import striqt.analysis as sa

load = sa.lib.io.decode_from_yaml_file


# %% !include


def test_glob_include_merges_in_sorted_filename_order(write_yaml):
    """the files are written out of name order. A key that 2 files set takes the
    value from the later name."""
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
    # YAML 1.1 leaves engineering-notation numbers as strings, so loop points
    # like 3750e6 reach _build_loop_points_dict untyped and it must coerce them
    # to the capture field type (xfail-audit #1)
    value = load(write_yaml('spec.yaml', f'v: {text}\n'))['v']
    assert value == expected
    assert type(value) is type(expected)


# %% repair_legacy_attrs


def legacy_store_dataset():
    """a dataset with the bandwidth encoded in `units`, no `noise_bandwidth` on the bin
    coordinate, the sample rate as the correlator `noise_bandwidth`, and a store-wide
    `noise_bandwidth` on the channel power outputs"""
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


def test_repair_legacy_attrs_removes_channel_power_bandwidth_attrs():
    """a store-wide `noise_bandwidth` and bandwidth-suffixed `long_name` on channel
    power outputs are removed; the bandwidth of those outputs is per capture"""
    ds = sa.lib.io.repair_legacy_attrs(legacy_store_dataset())

    names = (
        'channel_power_time_series',
        'channel_power_histogram',
        'channel_power_bin',
    )
    for name in names:
        assert 'noise_bandwidth' not in ds[name].attrs, name
        assert 'long_name' not in ds[name].attrs, name
    assert ds.channel_power_time_series.attrs['units'] == 'dBm'
    assert ds.channel_power_bin.attrs['units'] == 'dBm'


def test_repair_legacy_attrs_keeps_spectral_densities_and_current_stores():
    """dBm/Hz is a density, not a bandwidth label, and a second pass changes nothing"""
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
