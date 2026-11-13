"""striqt.analysis.lib.io: YAML decoding with `!include`, zarr store access and
allocation, and the attr repair of stores written with bandwidth-suffixed units."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import SWEEP_DIR

import striqt.analysis as sa
from striqt.analysis.lib import stores

load = sa.lib.io.decode_from_yaml_file
load_provenance = sa.lib.io.load_yaml_provenance
apply_edits = sa.lib.io.apply_yaml_edits
YAMLEdit = sa.lib.io.YAMLEdit

ZARR_MAJOR = int(zarr.__version__.split('.')[0])

# the keys of a group or array document in either zarr format
METADATA_BASENAMES = {'.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json'}


# %% !include


def test_glob_include_merges_in_sorted_filename_order(write_yaml):
    """a key set by 2 files of a glob include takes the value from the file with the
    later name, whatever the order in which the files were written."""
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


def test_include_resolves_against_the_including_file_under_a_relative_spec_path(
    write_yaml, tmp_path, monkeypatch
):
    """an `!include` path resolves against the including file when the spec path is
    relative to the working directory, at both levels of a nested include."""
    write_yaml('spec.yaml', 'plot: !include fragments/plot.yaml\n')
    write_yaml('fragments/plot.yaml', 'style: !include deeper/style.yaml\n')
    write_yaml('fragments/deeper/style.yaml', 'name: dark\n')
    monkeypatch.chdir(tmp_path.parent)
    assert load(f'{tmp_path.name}/spec.yaml') == {'plot': {'style': {'name': 'dark'}}}


def test_glob_include_merges_only_one_level_deep(write_yaml):
    write_yaml('sites/a.yaml', 'block:\n  x: 1\n  y: 1\n')
    write_yaml('sites/b.yaml', 'block:\n  y: 2\n')
    spec = write_yaml('spec.yaml', 'sites: !include "sites/*.yaml"\n')
    assert load(spec)['sites'] == {'block': {'y': 2}}


# (fragment files, spec path, spec text, expected document). A relative include path
# resolves against the including file, and ``{root}`` is the temporary root.
INCLUDE_PATHS = {
    'nested_relative_to_the_including_file': (
        {'frag/leaf.yaml': 'v: 1\n', 'frag/c.yaml': 'x: !include leaf.yaml\n'},
        'spec.yaml',
        'top: !include frag/c.yaml\n',
        {'top': {'x': {'v': 1}}},
    ),
    'parent_relative_glob_from_subdirectory': (
        {
            'sites/global.yaml': 'defaults: {a: 1}\n',
            'sites/radio.yaml': 'beef: {b: 2}\n',
        },
        'site/spec.yaml',
        'adjust: !include "../sites/*.yaml"\n',
        {'adjust': {'defaults': {'a': 1}, 'beef': {'b': 2}}},
    ),
    'absolute_inside_root': (
        {'frag.yaml': 'v: 1\n'},
        'site/spec.yaml',
        'top: !include {root}/frag.yaml\n',
        {'top': {'v': 1}},
    ),
    'absolute_outside_root': (
        {'elsewhere/frag.yaml': 'v: 1\n'},
        'site/spec.yaml',
        'top: !include {root}/elsewhere/frag.yaml\n',
        {'top': {'v': 1}},
    ),
}


@pytest.mark.parametrize('case', list(INCLUDE_PATHS), ids=list(INCLUDE_PATHS))
def test_include_path_resolution(write_yaml, tmp_path, case):
    fragments, spec_path, spec_text, expected = INCLUDE_PATHS[case]
    for path, text in fragments.items():
        write_yaml(path, text)
    spec = write_yaml(spec_path, spec_text.format(root=tmp_path))
    assert load(spec) == expected


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


# %% loader behavior


def test_flow_sequence_mapping_keys_become_tuples(write_yaml):
    spec = write_yaml('spec.yaml', 'lookup:\n  [0, 1]: 2\n  [1, 0]: 3\n')
    assert load(spec)['lookup'] == {(0, 1): 2, (1, 0): 3}


# %% load_yaml_provenance


@pytest.fixture
def sweeps(tmp_path) -> Path:
    """a copy of the fixture sweep tree, so that edits stay out of the repository"""
    dest = tmp_path / 'sweeps'
    shutil.copytree(SWEEP_DIR, dest, ignore=shutil.ignore_patterns('__pycache__'))
    return dest.resolve()


def as_lists(obj):
    """return `obj` with every tuple value read as a list, as the merged tree is"""
    if isinstance(obj, dict):
        return {k: as_lists(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_lists(v) for v in obj]
    return obj


def changed_lines(before: list[str], after: list[str]) -> list[int]:
    """the indexes at which 2 line lists of equal length differ"""
    assert len(before) == len(after)
    return [i for i, (a, b) in enumerate(zip(before, after)) if a != b]


@pytest.mark.parametrize(
    'name', ['site/site-cpu.yaml', 'synthetic.yaml', 'site/site-survey.yaml']
)
def test_provenance_tree_equals_the_pyyaml_decode(name):
    """the merged tree resolves includes, merge keys, and scalars (``3750e6`` as a
    string, ``1/8000`` as a string) as `decode_from_yaml_file` does."""
    path = SWEEP_DIR / name
    assert load_provenance(path).tree == as_lists(load(path))


def test_provenance_origin_names_the_file_that_set_each_key(sweeps):
    site = sweeps / 'site' / 'site-cpu.yaml'
    tone = sweeps / 'fragments' / 'source' / 'tone.yaml'
    tone_unsync = sweeps / 'fragments' / 'source' / 'tone_unsync.yaml'
    captures = sweeps / 'fragments' / 'captures' / 'tone.yaml'
    sites = tuple(sorted((sweeps / 'sites').glob('*.yaml')))

    prov = load_provenance(site)

    assert prov.root == site and prov.root.is_absolute()
    # the key set by both source fragments takes the origin of the later file
    assert prov.origin['/source/signal_trigger'] == (tone_unsync, '/signal_trigger')
    assert prov.origin['/source/master_clock_rate'] == (tone, '/master_clock_rate')
    assert prov.origin['/captures/0'] == (captures, '/0')
    assert prov.origin['/captures/0/gain'] == (captures, '/0/gain')
    assert prov.origin['/adjust_captures/defaults'] == (
        sweeps / 'sites' / 'global.yaml',
        '/defaults',
    )
    assert prov.regions['/source'] == (tone, tone_unsync)
    assert prov.regions['/adjust_captures'] == sites
    assert prov.regions['/captures'] == (captures,)
    assert next(iter(prov.files)) == site
    assert {tone, tone_unsync, captures, *sites} <= set(prov.files)
    assert all(p.is_absolute() for p in prov.files)


def test_provenance_resolves_merge_keys_in_the_owning_file(sweeps):
    spec = sweeps / 'synthetic.yaml'
    plot = sweeps / 'fragments' / 'plot' / 'synthetic.yaml'

    prov = load_provenance(spec)

    assert prov.origin['/plot_hint/data'] == (plot, '/data')
    assert prov.origin['/captures/0/snr'] == (spec, '/captures/0/snr')
    assert prov.tree['analysis']['power_spectral_density']['window'] == 'hann'
    assert prov.origin['/analysis/power_spectral_density/window'] == (
        spec,
        '/analysis/power_spectral_density/window',
    )
    assert prov.regions == {'/plot_hint': (plot,)}


# %% apply_yaml_edits


def test_set_rewrites_one_line_and_keeps_the_rest(sweeps):
    spec = sweeps / 'synthetic.yaml'
    before = spec.read_text().splitlines()

    written = apply_edits(
        load_provenance(spec), [YAMLEdit('/captures/0/snr', 'set', '70')]
    )

    after = spec.read_text().splitlines()
    assert written == (spec,)
    (index,) = changed_lines(before, after)
    assert before[index] == '    snr: 80 # dB'
    assert after[index] == '    snr: 70 # dB'
    assert 'plot_hint: !include fragments/plot/synthetic.yaml' in after
    assert load(spec)['captures'][0]['snr'] == 70


def test_set_through_an_include_writes_the_fragment_only(sweeps):
    site = sweeps / 'site' / 'site-cpu.yaml'
    fragment = sweeps / 'fragments' / 'captures' / 'tone.yaml'
    site_bytes = site.read_bytes()
    before = fragment.read_text().splitlines()

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/captures/0/gain', 'set', '-20')]
    )

    assert written == (fragment,)
    assert site.read_bytes() == site_bytes
    (index,) = changed_lines(before, fragment.read_text().splitlines())
    assert fragment.read_text().splitlines()[index] == '  gain: -20 # dB'
    assert load(site)['captures'][0]['gain'] == -20


def test_set_of_a_new_key_in_a_merged_region_lands_in_the_last_file(sweeps):
    site = sweeps / 'site' / 'site-cpu.yaml'
    tone = sweeps / 'fragments' / 'source' / 'tone.yaml'
    tone_unsync = sweeps / 'fragments' / 'source' / 'tone_unsync.yaml'
    tone_bytes = tone.read_bytes()

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/source/warmup_sweeps', 'set', '2')]
    )

    assert written == (tone_unsync,)
    assert tone.read_bytes() == tone_bytes
    assert tone_unsync.read_text().endswith('warmup_sweeps: 2\n')
    assert load(site)['source']['warmup_sweeps'] == 2


def test_delete_removes_the_key_from_the_fragment(sweeps):
    site = sweeps / 'site' / 'site-cpu.yaml'
    fragment = sweeps / 'fragments' / 'captures' / 'tone.yaml'
    before = fragment.read_text().splitlines()

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/captures/0/backend_sample_rate', 'delete')]
    )

    after = fragment.read_text().splitlines()
    assert written == (fragment,)
    assert len(after) == len(before) - 1
    assert 'backend_sample_rate' not in load(site)['captures'][0]


def test_insert_appends_a_capture_to_the_fragment_sequence(sweeps):
    site = sweeps / 'site' / 'site-cpu.yaml'
    fragment = sweeps / 'fragments' / 'captures' / 'tone.yaml'
    site_bytes = site.read_bytes()
    capture = 'port: 1\ncenter_frequency: 3900e6 # Hz\n'

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/captures/-', 'insert', capture)]
    )

    assert written == (fragment,)
    assert site.read_bytes() == site_bytes
    assert fragment.read_text().endswith('- port: 1\n  center_frequency: 3900e6 # Hz\n')
    captures = load(site)['captures']
    assert len(captures) == 2
    assert captures[1] == {'port': 1, 'center_frequency': '3900e6'}


def test_float_text_and_flow_sequence_are_written_as_typed(sweeps):
    spec = sweeps / 'synthetic.yaml'

    apply_edits(
        load_provenance(spec),
        [
            YAMLEdit('/source/master_clock_rate', 'set', '3750e6'),
            YAMLEdit('/captures/1/external_lo_frequency', 'set', '[10.0e9, 11.0e9]'),
        ],
    )

    lines = spec.read_text().splitlines()
    assert '  master_clock_rate: 3750e6 # Hz' in lines
    assert '    external_lo_frequency: [10.0e9, 11.0e9] # one per port' in lines
    assert load(spec)['source']['master_clock_rate'] == '3750e6'


@pytest.mark.parametrize(
    'edit,exc',
    [
        (YAMLEdit('/captures/5/gain', 'set', '0'), KeyError),
        (YAMLEdit('/nothing/here', 'delete'), KeyError),
        (YAMLEdit('/captures/0/gain', 'replace', '0'), ValueError),
        (YAMLEdit('/captures/0/gain', 'set', None), ValueError),
        (YAMLEdit('/captures/0/gain', 'insert', '0'), TypeError),
    ],
    ids=['missing_element', 'missing_key', 'unknown_op', 'no_value', 'not_a_sequence'],
)
def test_edit_errors_write_nothing(sweeps, edit, exc):
    site = sweeps / 'site' / 'site-cpu.yaml'
    fragment = sweeps / 'fragments' / 'captures' / 'tone.yaml'
    fragment_bytes = fragment.read_bytes()

    with pytest.raises(exc):
        apply_edits(load_provenance(site), [edit])

    assert fragment.read_bytes() == fragment_bytes


def test_delete_of_a_merge_key_value_raises(sweeps):
    """``window`` of `power_spectral_density` comes from the ``<<: *spectrogram``
    alias, so it is not a key of that mapping to remove."""
    spec = sweeps / 'synthetic.yaml'
    with pytest.raises(KeyError, match='merge key'):
        apply_edits(
            load_provenance(spec),
            [YAMLEdit('/analysis/power_spectral_density/window', 'delete')],
        )


# %% repair_legacy_attrs


def legacy_store_dataset():
    """build a dataset with the bandwidth encoded in `units`, the bandwidth as a
    suffix of `long_name` and `standard_name`, no `noise_bandwidth` on the bin
    coordinate, the sample rate as the correlator `noise_bandwidth`, and a store-wide
    `noise_bandwidth` on the channel power outputs."""
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
            'power_spectral_density': (
                ('capture', 'baseband_frequency'),
                np.zeros((2, 3), 'float32'),
                {
                    'standard_name': 'PSD',
                    'long_name': 'Power spectral density (dBm/360 kHz)',
                    'units': 'dBm',
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
            'channel_power': (
                ('capture',),
                np.zeros(2, 'float32'),
                {
                    'standard_name': 'Channel power (dBW/10 MHz)',
                    'long_name': 'Mean channel power (dBW/10 MHz)',
                    'units': 'dBW',
                    'noise_bandwidth': 10e6,
                },
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


# the attrs of each variable of `legacy_store_dataset` after the repair:
# - `units` of the form ``'<unit>/<bandwidth>'`` loses the bandwidth
# - a ``' (<unit>/<bandwidth>)'`` suffix is cut from `long_name` and `standard_name`
# - a bin coordinate without a `noise_bandwidth` takes the one of the histogram on its
#   dimension
# - a ratio coordinate and histogram have no `noise_bandwidth`
# - a correlator `noise_bandwidth` equal to the sample rate becomes the subcarrier
#   spacing
# - the channel power and IQ outputs lose the store-wide `noise_bandwidth` and the
#   bandwidth-suffixed `long_name`, because that bandwidth is set per capture
# - a density unit is kept
REPAIRED_ATTRS = {
    'spectrogram': {
        'standard_name': 'PSD',
        'long_name': 'Power Spectral Density',
        'units': 'dBm',
        'noise_bandwidth': 360e3,
    },
    'power_spectral_density': {
        'standard_name': 'PSD',
        'long_name': 'Power spectral density',
        'units': 'dBm',
        'noise_bandwidth': 360e3,
    },
    'spectrogram_histogram': {
        'standard_name': 'Fraction of counts',
        'noise_bandwidth': 360e3,
    },
    'spectrogram_ratio_histogram': {'standard_name': 'Fraction of counts'},
    'channel_power_time_series': {'standard_name': 'Channel Power', 'units': 'dBm'},
    'channel_power_histogram': {'standard_name': 'Fraction of channel power readings'},
    'iq_waveform': {'standard_name': 'IQ waveform', 'units': '√mW'},
    'cellular_5g_pss_correlation': {
        'standard_name': 'PSS Cross-Covariance',
        'units': '√mW',
        'noise_bandwidth': 30e3,
        'sample_rate': 7.68e6,
        'subcarrier_spacing': 30e3,
    },
    'system_noise': {
        'name': 'Sensor system noise spectral density',
        'units': 'dBm/Hz',
        'standard_name': 'System noise',
    },
    'channel_power': {
        'standard_name': 'Channel power',
        'long_name': 'Mean channel power',
        'units': 'dBW',
        'noise_bandwidth': 10e6,
    },
    'spectrogram_power_bin': {
        'standard_name': 'Spectrogram bin power',
        'units': 'dBm',
        'noise_bandwidth': 360e3,
    },
    'channel_power_bin': {'standard_name': 'Channel power', 'units': 'dBm'},
    'spectrogram_ratio_power_bin': {
        'standard_name': 'Spectrogram cross-channel power ratio',
        'units': 'dB',
    },
}


@pytest.fixture(scope='module')
def repaired_legacy_dataset() -> xr.Dataset:
    return sa.lib.io.repair_legacy_attrs(legacy_store_dataset())


def test_repair_legacy_attrs_covers_every_variable(repaired_legacy_dataset):
    assert set(repaired_legacy_dataset.variables) == set(REPAIRED_ATTRS)


@pytest.mark.parametrize('variable', list(REPAIRED_ATTRS), ids=list(REPAIRED_ATTRS))
def test_repair_legacy_attrs(repaired_legacy_dataset, variable):
    assert dict(repaired_legacy_dataset[variable].attrs) == REPAIRED_ATTRS[variable]


def test_repair_legacy_attrs_is_idempotent(repaired_legacy_dataset):
    before = {
        name: dict(repaired_legacy_dataset[name].attrs)
        for name in repaired_legacy_dataset.variables
    }
    sa.lib.io.repair_legacy_attrs(repaired_legacy_dataset)
    after = {
        name: dict(repaired_legacy_dataset[name].attrs)
        for name in repaired_legacy_dataset.variables
    }
    assert after == before


def test_repair_legacy_attrs_keeps_the_full_scale_and_arbitrary_families():
    """units of the FS and a.u. families match no repair condition: no bandwidth
    suffix, no ``'√mW'`` correlator, and a bin coordinate that already names its
    `noise_bandwidth`."""
    bins = np.array([-np.inf, -100.0, -99.0, np.inf])
    ds = xr.Dataset(
        {
            'channel_power_time_series': (
                ('capture', 'time_elapsed'),
                np.zeros((2, 3), 'float32'),
                {'standard_name': 'Channel power', 'units': 'dBFS'},
            ),
            'iq_waveform': (
                ('capture', 'iq_index'),
                np.zeros((2, 3), 'complex64'),
                {'standard_name': 'IQ waveform', 'units': '√FS'},
            ),
            'cellular_cyclic_autocorrelation': (
                ('capture', 'cyclic_sample_lag'),
                np.zeros((2, 3), 'float32'),
                {'standard_name': 'Cyclic Autocovariance', 'units': 'a.u.'},
            ),
            'spectrogram': (
                ('capture', 'spectrogram_time'),
                np.zeros((2, 3), 'float32'),
                {
                    'long_name': 'Power spectral density',
                    'units': 'dB(a.u.)',
                    'noise_bandwidth': 360e3,
                },
            ),
            'spectrogram_histogram': (
                ('capture', 'spectrogram_power_bin'),
                np.zeros((2, 4), 'float32'),
                {'standard_name': 'Fraction of counts', 'noise_bandwidth': 360e3},
            ),
        },
        coords={
            'spectrogram_power_bin': (
                'spectrogram_power_bin',
                bins,
                {
                    'long_name': 'Spectrogram bin power',
                    'units': 'dBFS',
                    'noise_bandwidth': 360e3,
                },
            )
        },
    )
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
    """return a dataset of 2 registered measurements on a fixed coordinate `x` and a
    datetime `start_time` coordinate, with 0 rows along `capture`.

    `channel_power_time_series` is registered with `store_compressed=True` and
    `iq_waveform` with `store_compressed=False`.
    """
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
    """allocate `zero_row_template` at `path` through a metadata-staging store and
    return the store with the allocated arrays."""
    store = stores.staged_metadata_store(sa.lib.io.open_store(path, mode='w'))
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

    if suffix == '.zip':
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        assert len(names) == len(set(names)), 'a zip entry written twice'

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
