"""the docstring gate: no personal pronouns or "whose", and no `.lib` paths in public
docstrings; and the structure of the generated measurement docstrings.

The AST scan imports nothing from `striqt`; the registry cell imports
`striqt.analysis` inside each test.
"""

import ast
import pathlib
import re
from collections.abc import Iterator

SRC = pathlib.Path(__file__).parent.parent / 'src' / 'striqt'

# "I" and "us" are not scanned: "I/Q" and the microsecond abbreviation
PRONOUN = re.compile(r'\b(we|our|you|your|they|their|them|whose)\b', re.IGNORECASE)
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
                rules.append('personal pronoun or "whose"')
            if public and LIB_PATH.search(doc):
                rules.append('striqt.<pkg>.lib path in a public docstring')
            if rules:
                hits[f'{rel}::{qualname}'] = '; '.join(rules)
    return hits


def test_docstrings_are_clean() -> None:
    hits = scan()
    lines = [f'{k}: {v}' for k, v in sorted(hits.items())]
    assert not hits, 'docstrings that break a prose rule:\n' + '\n'.join(lines)


# %% registry docstrings


def test_registered_long_names_are_sentence_case_titles(subtests) -> None:
    """a title starts with a capital letter or a digit, contains no underscore, and
    has no surrounding whitespace"""
    import striqt.analysis as sa

    for info in sa.registry.values():
        with subtests.test(msg=info.name):
            title = info.attrs['long_name']
            assert '_' not in title
            assert title[0].isupper() or title[0].isdigit()
            assert title == title.strip()


def test_registered_measurement_docstrings_render_the_output_section(subtests) -> None:
    """every builtin measurement writes the `Output:` section with the placeholders,
    the registry renders the dtype, every dimension, and every coordinate factory into
    the section, and the prose above the section names no attr"""
    import striqt.analysis as sa
    from striqt.analysis.lib import dataarrays

    for info in sa.registry.values():
        with subtests.test(msg=info.name):
            doc = info.func.__doc__

            assert doc is not None
            assert 'Output:' in doc
            for placeholder in ('{args}', '{dtype}', '{dims}', '{coords}', '{attrs}'):
                assert placeholder not in doc
            assert f'- Array dtype: {info.dtype}' in doc
            dims = ('port',) + dataarrays.measurement_dims(info)
            assert '- Dimensions: ' + ', '.join(f'`{dim}`' for dim in dims) in doc
            for factory in info.coord_factories:
                name = sa.registry.coordinates[factory].name
                assert f'* `{name}` (' in doc
            if not info.coord_factories:
                assert '- Coordinates:' not in doc
            if 'units' in info.attrs:
                assert '* `units`: ' in doc
            if '- Attributes:' in doc:
                assert re.search(r'- Attributes:\s+\* ', doc)
            prose = doc[: doc.index('Output:')]
            assert ' attr ' not in prose and ' attrs ' not in prose


def test_registered_coordinate_factories_have_a_docstring(subtests) -> None:
    """the first paragraph of the factory docstring is the description of the
    coordinate in every measurement docstring that lists the coordinate, so the
    paragraph is a noun phrase rather than a `Returns` sentence"""
    import inspect

    import striqt.analysis as sa

    for factory, info in sa.registry.coordinates.items():
        with subtests.test(msg=info.name):
            doc = inspect.getdoc(factory)

            assert doc, f'{factory.__name__} has no docstring'
            first = doc.split('\n\n', 1)[0]
            assert not first.lower().startswith(('return ', 'returns '))
