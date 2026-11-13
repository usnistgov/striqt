"""striqt.analysis.lib.yaml: YAML decoding with `!include`, the provenance tree of a
spec file and the edits written back into its files, JSON pointers, and flow text."""

from __future__ import annotations


import pytest
import yaml
from conftest import SWEEP_DIR

import striqt.analysis as sa
from striqt.analysis.lib import yaml as spec_yaml

load = spec_yaml.decode_from_yaml_file
load_provenance = spec_yaml.load_yaml_provenance
apply_edits = spec_yaml.apply_yaml_edits
YAMLEdit = spec_yaml.YAMLEdit


# %% merge_included


def test_merge_included_passes_one_document_through():
    doc = {'x': (1, 2)}
    assert spec_yaml.merge_included([doc]) is doc


def test_merge_included_updates_mappings_in_order():
    merged = spec_yaml.merge_included([{'x': 1, 'y': 1}, {'y': 2, 'z': 2}, {'z': 3}])
    assert merged == {'x': 1, 'y': 2, 'z': 3}
    assert list(merged) == ['x', 'y', 'z']


def test_merge_included_concatenates_sequences():
    assert spec_yaml.merge_included([(1, 2), [3], ()]) == [1, 2, 3]


@pytest.mark.parametrize(
    'contents',
    [[{'x': 1}, [3]], [[3], {'x': 1}], [1, 2]],
    ids=['mapping_then_sequence', 'sequence_then_mapping', 'scalars'],
)
def test_merge_included_rejects_mixed_or_scalar_documents(contents):
    with pytest.raises(TypeError, match='all mappings or all sequences'):
        spec_yaml.merge_included(contents)


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


# %% JSON pointers


@pytest.mark.parametrize(
    'token,escaped',
    [('plain', 'plain'), ('a/b', 'a~1b'), ('a~b', 'a~0b'), ('~1', '~01'), ('', '')],
    ids=['plain', 'slash', 'tilde', 'tilde_digit', 'empty'],
)
def test_pointer_token_escaping_round_trips(token, escaped):
    """RFC 6901 §4 unescapes ``~1`` before ``~0``, so a key that contains ``~1``
    escapes to ``~01`` and reads back unchanged."""
    assert spec_yaml.escape_pointer_token(token) == escaped
    assert spec_yaml.unescape_pointer_token(escaped) == token


def test_json_pointer_joins_escaped_tokens():
    assert spec_yaml.json_pointer([]) == ''
    assert spec_yaml.json_pointer(['captures', 0, 'a/b']) == '/captures/0/a~1b'
    assert [
        spec_yaml.unescape_pointer_token(t)
        for t in spec_yaml.json_pointer(['x~y', 'z']).split('/')[1:]
    ] == ['x~y', 'z']


# %% yaml_flow_text


@pytest.mark.parametrize(
    ('value', 'text'),
    [
        (70, '70'),
        ('3750e6', '3750e6'),
        ('yes', "'yes'"),
        (None, 'null'),
        ([0, 1], '[0, 1]'),
        ({'a': 1, 'b': [True, 'x']}, '{a: 1, b: [true, x]}'),
    ],
)
def test_yaml_flow_text_reads_back_as_the_value(value, text):
    assert spec_yaml.yaml_flow_text(value) == text
    assert yaml.safe_load(text) == value


# %% tree_edits


def test_tree_edits_touch_only_the_differences():
    current = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': [1, 2], 'f': [1, 2, 3]}
    target = {'a': 1, 'b': {'c': 2, 'd': 4, 'g': 5}, 'f': [1, 2], 'h': 'x'}
    edits = spec_yaml.tree_edits(current, target, '/s')
    assert [(e.pointer, e.op, e.value_text) for e in edits] == [
        ('/s/e', 'delete', None),
        ('/s/b/d', 'set', '4'),
        ('/s/b/g', 'set', '5'),
        ('/s/f', 'set', '[1, 2]'),
        ('/s/h', 'set', 'x'),
    ]


def test_tree_edits_of_equal_trees_are_empty():
    tree = {'a': [1, {'b': 'c'}]}
    assert spec_yaml.tree_edits(tree, dict(tree)) == []


def test_tree_edits_escape_keys_and_index_equal_length_lists():
    current = {'a/b': [{'x': 1}, {'x': 2}]}
    target = {'a/b': [{'x': 1}, {'x': 3}]}
    assert spec_yaml.tree_edits(current, target) == [YAMLEdit('/a~1b/1/x', 'set', '3')]


# %% load_yaml_provenance


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
    assert load_provenance(path).tree == sa.specs.helpers.unfreeze(load(path))


def test_provenance_origin_names_the_file_that_set_each_key(sweeps_copy):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    tone = sweeps_copy / 'fragments' / 'source' / 'tone.yaml'
    tone_unsync = sweeps_copy / 'fragments' / 'source' / 'tone_unsync.yaml'
    captures = sweeps_copy / 'fragments' / 'captures' / 'tone.yaml'
    sites = tuple(sorted((sweeps_copy / 'sites').glob('*.yaml')))

    prov = load_provenance(site)

    assert prov.root == site and prov.root.is_absolute()
    # the key set by both source fragments takes the origin of the later file
    assert prov.origin['/source/signal_trigger'] == (tone_unsync, '/signal_trigger')
    assert prov.origin['/source/master_clock_rate'] == (tone, '/master_clock_rate')
    assert prov.origin['/captures/0'] == (captures, '/0')
    assert prov.origin['/captures/0/gain'] == (captures, '/0/gain')
    assert prov.origin['/adjust_captures/defaults'] == (
        sweeps_copy / 'sites' / 'global.yaml',
        '/defaults',
    )
    assert prov.regions['/source'] == (tone, tone_unsync)
    assert prov.regions['/adjust_captures'] == sites
    assert prov.regions['/captures'] == (captures,)
    assert next(iter(prov.files)) == site
    assert {tone, tone_unsync, captures, *sites} <= set(prov.files)
    assert all(p.is_absolute() for p in prov.files)


def test_provenance_resolves_merge_keys_in_the_owning_file(sweeps_copy):
    spec = sweeps_copy / 'synthetic.yaml'
    plot = sweeps_copy / 'fragments' / 'plot' / 'synthetic.yaml'

    prov = load_provenance(spec)

    assert prov.origin['/plot_hint/data'] == (plot, '/data')
    assert prov.origin['/captures/0/snr'] == (spec, '/captures/0/snr')
    assert prov.tree['analysis']['power_spectral_density']['window'] == 'hann'
    assert prov.origin['/analysis/power_spectral_density/window'] == (
        spec,
        '/analysis/power_spectral_density/window',
    )
    assert prov.regions == {'/plot_hint': (plot,)}


def test_provenance_of_concatenated_includes_offsets_the_element_origins(write_yaml):
    a = write_yaml('a.yaml', '- 1\n- 2\n')
    b = write_yaml('b.yaml', '- 3\n')
    spec = write_yaml('spec.yaml', 'items: !include [a.yaml, b.yaml]\n')

    prov = load_provenance(spec)

    assert prov.tree == {'items': [1, 2, 3]}
    assert prov.origin['/items/1'] == (a.resolve(), '/1')
    assert prov.origin['/items/2'] == (b.resolve(), '/0')
    assert prov.regions['/items'] == (a.resolve(), b.resolve())


# %% apply_yaml_edits


def test_set_rewrites_one_line_and_keeps_the_rest(sweeps_copy):
    spec = sweeps_copy / 'synthetic.yaml'
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


def test_set_through_an_include_writes_the_fragment_only(sweeps_copy):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    fragment = sweeps_copy / 'fragments' / 'captures' / 'tone.yaml'
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


def test_set_of_a_new_key_in_a_merged_region_lands_in_the_last_file(sweeps_copy):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    tone = sweeps_copy / 'fragments' / 'source' / 'tone.yaml'
    tone_unsync = sweeps_copy / 'fragments' / 'source' / 'tone_unsync.yaml'
    tone_bytes = tone.read_bytes()

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/source/warmup_sweeps', 'set', '2')]
    )

    assert written == (tone_unsync,)
    assert tone.read_bytes() == tone_bytes
    assert tone_unsync.read_text().endswith('warmup_sweeps: 2\n')
    assert load(site)['source']['warmup_sweeps'] == 2


def test_delete_removes_the_key_from_the_fragment(sweeps_copy):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    fragment = sweeps_copy / 'fragments' / 'captures' / 'tone.yaml'
    before = fragment.read_text().splitlines()

    written = apply_edits(
        load_provenance(site), [YAMLEdit('/captures/0/backend_sample_rate', 'delete')]
    )

    after = fragment.read_text().splitlines()
    assert written == (fragment,)
    assert len(after) == len(before) - 1
    assert 'backend_sample_rate' not in load(site)['captures'][0]


def test_insert_appends_a_capture_to_the_fragment_sequence(sweeps_copy):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    fragment = sweeps_copy / 'fragments' / 'captures' / 'tone.yaml'
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


def test_float_text_and_flow_sequence_are_written_as_typed(sweeps_copy):
    spec = sweeps_copy / 'synthetic.yaml'

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
def test_edit_errors_write_nothing(sweeps_copy, edit, exc):
    site = sweeps_copy / 'site' / 'site-cpu.yaml'
    fragment = sweeps_copy / 'fragments' / 'captures' / 'tone.yaml'
    fragment_bytes = fragment.read_bytes()

    with pytest.raises(exc):
        apply_edits(load_provenance(site), [edit])

    assert fragment.read_bytes() == fragment_bytes


def test_delete_of_a_merge_key_value_raises(sweeps_copy):
    """``window`` of `power_spectral_density` comes from the ``<<: *spectrogram``
    alias, so it is not a key of that mapping to remove."""
    spec = sweeps_copy / 'synthetic.yaml'
    with pytest.raises(KeyError, match='merge key'):
        apply_edits(
            load_provenance(spec),
            [YAMLEdit('/analysis/power_spectral_density/window', 'delete')],
        )
