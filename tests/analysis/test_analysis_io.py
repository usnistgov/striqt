"""striqt.analysis.lib.io.decode_from_yaml_file: `!include` globs, lists and nesting,
flow-sequence keys, scalar typing"""

from __future__ import annotations

import functools
import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

import striqt.analysis as sa

load = sa.lib.io.decode_from_yaml_file
_example_dirs = itertools.count()


def _dump(d: dict) -> str:
    return ''.join(f'{k}: {v}\n' for k, v in d.items())


# %% !include


@given(
    entries=st.lists(
        st.tuples(
            st.from_regex(r'\A[a-z]{1,6}\Z'),
            st.dictionaries(st.sampled_from('abcd'), st.integers(), min_size=1),
        ),
        min_size=1,
        max_size=4,
        unique_by=lambda e: e[0],
    )
)
def test_glob_include_merges_in_sorted_filename_order(write_yaml, entries):
    sub = f'case{next(_example_dirs)}'
    for name, d in entries:
        write_yaml(f'{sub}/sites/{name}.yaml', _dump(d))
    spec = write_yaml(f'{sub}/spec.yaml', 'sites: !include "sites/*.yaml"\n')

    expected = functools.reduce(
        lambda a, b: {**a, **b}, [d for _, d in sorted(entries)]
    )
    assert load(spec)['sites'] == expected


def test_list_include_merges_dicts_with_later_files_winning(write_yaml):
    write_yaml('a.yaml', 'x: 1\ny: 1\n')
    write_yaml('b.yaml', 'y: 2\n')
    spec = write_yaml('spec.yaml', 'source: !include [a.yaml, b.yaml]\n')
    assert load(spec)['source'] == {'x': 1, 'y': 2}


def test_glob_include_merges_only_one_level_deep(write_yaml):
    write_yaml('sites/a.yaml', 'block:\n  x: 1\n  y: 1\n')
    write_yaml('sites/b.yaml', 'block:\n  y: 2\n')
    spec = write_yaml('spec.yaml', 'sites: !include "sites/*.yaml"\n')
    assert load(spec)['sites'] == {'block': {'y': 2}}


def test_list_include_applies_files_in_listed_order(write_yaml):
    write_yaml('a.yaml', 'y: 1\n')
    write_yaml('b.yaml', 'x: 1\ny: 2\n')
    spec = write_yaml('spec.yaml', 'source: !include [b.yaml, a.yaml]\n')
    assert load(spec)['source'] == {'x': 1, 'y': 1}


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


def test_absolute_include_inside_root(write_yaml):
    frag = write_yaml('frag.yaml', 'v: 1\n')
    spec = write_yaml('spec.yaml', f'top: !include {frag}\n')
    assert load(spec)['top'] == {'v': 1}


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason='_expand_paths rewrites every path relative to the top-level directory, '
    'which fails for an absolute include outside it',
)
def test_absolute_include_outside_root(write_yaml):
    frag = write_yaml('elsewhere/frag.yaml', 'v: 1\n')
    spec = write_yaml('site/spec.yaml', f'top: !include {frag}\n')
    assert load(spec)['top'] == {'v': 1}


def test_nested_include_in_the_same_directory(write_yaml):
    write_yaml('leaf.yaml', 'v: 1\n')
    write_yaml('frag.yaml', 'x: !include leaf.yaml\n')
    spec = write_yaml('spec.yaml', 'top: !include frag.yaml\n')
    assert load(spec)['top'] == {'x': {'v': 1}}


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason='nested includes are globbed relative to the top-level file but opened '
    'relative to the including fragment',
)
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
