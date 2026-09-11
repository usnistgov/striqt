"""the `extensions:` block: importing out-of-tree binding modules while reading a spec"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from conftest import SWEEP_DIR

import striqt.sensor as ss

INLINE_SPEC = """
sensor_binding: single_tone
extensions:
  import_path: {import_path}
  import_name: {import_name}
source:
  master_clock_rate: 125e6
  num_rx_ports: 2
captures:
  - port: 0
    duration: 1e-3
    sample_rate: 1e6
"""


def _write_spec(write_yaml, relpath, import_path, import_name='extensions'):
    return write_yaml(
        relpath, INLINE_SPEC.format(import_path=import_path, import_name=import_name)
    )


def test_fixture_extension_module_registers_site_bindings():
    import site_strategies as S

    for name in ('site_single_tone', 'site_survey', 'site_single_tone_calibration'):
        assert name in ss.lib.bindings.registry
    assert Path(S.EXT.__file__).resolve().parent == (SWEEP_DIR / 'src').resolve()


def test_import_path_is_relative_to_the_spec_directory(
    write_yaml, isolated_extension_import, caplog, tmp_path
):
    write_yaml('ext/extensions.py', "MARKER = 'tmp'\n")
    spec = _write_spec(write_yaml, 'sub/spec.yaml', '../ext')

    ss.read_yaml_spec(spec)

    assert sys.modules['extensions'].MARKER == 'tmp'
    assert Path(sys.path[0]).resolve() == (tmp_path / 'ext').resolve()
    assert 'did not bind a sensor' in caplog.text


def test_import_name_null_only_extends_sys_path(write_yaml, isolated_extension_import):
    write_yaml('ext/placeholder.txt', '')
    spec = _write_spec(write_yaml, 'sub/spec.yaml', '../ext', import_name='null')

    ss.read_yaml_spec(spec)

    assert Path(sys.path[0]).resolve() == (spec.parent / '../ext').resolve()
    assert 'extensions' not in sys.modules


def test_rereading_a_spec_warns_that_nothing_new_was_bound(site_spec_path, caplog):
    # documented as current: the binding count is compared before and after every
    # import, so the second read of any spec that shares a module logs this warning
    ss.read_yaml_spec(site_spec_path)
    assert 'did not bind a sensor' in caplog.text


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason='_import_extensions_from_spec checks that the root directory exists but '
    'not that root/import_path does',
)
def test_missing_import_path_raises_file_not_found(
    write_yaml, isolated_extension_import
):
    spec = _write_spec(write_yaml, 'sub/spec.yaml', 'nope')
    with pytest.raises(FileNotFoundError, match='nope'):
        ss.read_yaml_spec(spec)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='importlib.import_module returns the cached module from the first spec, '
    'so a second sensor directory with the same import_name is never imported',
)
def test_second_directory_with_the_same_import_name_is_imported(
    write_yaml, isolated_extension_import
):
    specs = {}
    for tag in 'ab':
        write_yaml(f'{tag}/ext/extensions.py', f"MARKER = '{tag}'\n")
        specs[tag] = _write_spec(write_yaml, f'{tag}/sub/spec.yaml', '../ext')

    ss.read_yaml_spec(specs['a'])
    assert sys.modules['extensions'].MARKER == 'a'
    ss.read_yaml_spec(specs['b'])
    assert sys.modules['extensions'].MARKER == 'b'
