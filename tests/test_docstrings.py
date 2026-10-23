"""the docstring gate: no pronouns for code and no `.lib` paths in public docstrings"""

import ast
import pathlib
import re
from collections.abc import Iterator

import pytest

SRC = pathlib.Path(__file__).parent.parent / 'src' / 'striqt'
BASELINE = pathlib.Path(__file__).parent / 'docstring-baseline.txt'

PRONOUN = re.compile(r'\b[Ii]ts?\b')
LIB_PATH = re.compile(r'\bstriqt\.\w+\.lib\b')


def prose_of(doc: str) -> str:
    """return `doc` without doctest lines and without backticked spans."""
    lines = [
        line
        for line in doc.splitlines()
        if not line.lstrip().startswith(('>>>', '...'))
    ]
    text = '\n'.join(lines)
    text = re.sub(r'``[^`]*``', '', text)
    return re.sub(r'`[^`]*`', '', text)


def iter_docstrings(path: pathlib.Path) -> Iterator[tuple[str, str]]:
    """yield ``(qualname, docstring)`` for the module and each class and function in `path`."""
    tree = ast.parse(path.read_text())
    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc is not None:
        yield '<module>', module_doc

    def visit(node: ast.AST, prefix: str) -> Iterator[tuple[str, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = prefix + child.name
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    yield qualname, doc
                yield from visit(child, qualname + '.')

    yield from visit(tree, '')


def scan() -> dict[str, str]:
    """return the entries ``'<relpath>::<qualname>'`` of docstrings that fail a rule, with the rule."""
    hits: dict[str, str] = {}
    for path in sorted(SRC.rglob('*.py')):
        rel = path.relative_to(SRC.parent.parent).as_posix()
        public = 'lib' not in path.relative_to(SRC).parts
        for qualname, doc in iter_docstrings(path):
            rules = []
            if PRONOUN.search(prose_of(doc)):
                rules.append('pronoun for code ("it"/"its")')
            if public and LIB_PATH.search(doc):
                rules.append('striqt.<pkg>.lib path in a public docstring')
            if rules:
                hits[f'{rel}::{qualname}'] = '; '.join(rules)
    return hits


def read_baseline() -> set[str]:
    return {
        line.strip()
        for line in BASELINE.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    }


@pytest.fixture(scope='module')
def hits() -> dict[str, str]:
    return scan()


def test_no_new_docstring_hits(hits: dict[str, str]) -> None:
    new = {k: v for k, v in hits.items() if k not in read_baseline()}
    lines = [f'{k}: {v}' for k, v in sorted(new.items())]
    assert not new, 'docstrings that break a prose rule:\n' + '\n'.join(lines)


def test_docstring_baseline_is_current(hits: dict[str, str]) -> None:
    stale = sorted(read_baseline() - set(hits))
    assert not stale, (
        f'remove these entries from {BASELINE.name}, which no longer hit:\n'
        + '\n'.join(stale)
    )
