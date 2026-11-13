"""``docs/gen_api.py`` writes one page for every ``api/`` entry of the zensical nav"""

import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]


def nav_pages(node) -> set:
    """return the ``.md`` paths named anywhere under a zensical nav entry"""
    if isinstance(node, str):
        return {node}
    if isinstance(node, dict):
        node = list(node.values())
    pages = set()
    for item in node:
        pages |= nav_pages(item)
    return pages


def find_nav(node):
    """return the first ``nav`` list under any table of a zensical config"""
    if isinstance(node, dict):
        if 'nav' in node:
            return node['nav']
        for value in node.values():
            found = find_nav(value)
            if found is not None:
                return found
    return None


# %% main
@pytest.mark.skipif(tomllib is None, reason='tomllib requires python >= 3.11')
def test_generator_writes_every_api_page_of_the_nav(tmp_path) -> None:
    # a fresh interpreter isolates the generator's imports from the test session
    result = subprocess.run(
        [sys.executable, str(ROOT / 'docs' / 'gen_api.py'), '--out', str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with open(ROOT / 'zensical.toml', 'rb') as stream:
        config = tomllib.load(stream)
    nav = find_nav(config)
    assert nav is not None, 'zensical.toml has no nav'
    expected = {p[len('api/') :] for p in nav_pages(nav) if p.startswith('api/')}
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.md')}
    assert expected
    assert expected <= written, expected - written
    for page in written:
        assert ':::' in (tmp_path / page).read_text()
