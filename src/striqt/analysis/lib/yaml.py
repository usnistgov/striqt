"""YAML spec files: the ``!include`` decoder, and the provenance tree through which
`apply_yaml_edits` writes edits back into the files that hold each value.

Scalars resolve under the PyYAML YAML 1.1 rules in both loaders (``3750e6`` is a
string, ``yes`` is a bool); the round-trip loader copies the PyYAML resolver tables,
so the merged tree equals the `decode_from_yaml_file` result with tuples read as
lists.
"""

from __future__ import annotations as __

import contextlib
import dataclasses
import functools
import os
import shutil
import tempfile
import threading
import typing
from collections.abc import Sequence
from pathlib import Path

from .. import specs
from . import util

if typing.TYPE_CHECKING:
    import ruamel.yaml
    import ruamel.yaml.comments
    import ruamel.yaml.constructor
    import ruamel.yaml.representer
    import ruamel.yaml.resolver
    import ruamel.yaml.scalarbool
    import yaml
    import yaml.resolver

else:
    yaml = util.lazy_import('yaml')
    util.lazy_import('ruamel.yaml')
    import ruamel

_INCLUDE_TAG = '!include'

EditOp = typing.Literal['set', 'delete', 'insert']
_EDIT_OPS = typing.get_args(EditOp)


def merge_included(contents: list[typing.Any]) -> typing.Any:
    """merge the documents of the files matched by one ``!include`` tag.

    Args:
        contents: the decoded documents, in match order

    Returns:
        the only document when `contents` has 1 item; a `dict` updated from each
        mapping in order, so a key set in several files takes the value from the
        last; or a `list` of the sequences concatenated in order

    Raises:
        TypeError: when `contents` mixes mappings and sequences, or holds a scalar
    """
    if len(contents) == 1:
        return contents[0]

    if all(isinstance(c, dict) for c in contents):
        merged: dict[typing.Any, typing.Any] = {}
        for mapping in contents:
            merged.update(mapping)
        return merged

    if all(isinstance(c, (list, tuple)) for c in contents):
        concatenated: list[typing.Any] = []
        for sequence in contents:
            concatenated.extend(sequence)
        return concatenated

    raise TypeError(
        f'contents of several {_INCLUDE_TAG} files must be '
        'either all mappings or all sequences'
    )


# %% PyYAML loader


class _YAMLFrozenLoader(yaml.SafeLoader):
    def _construct_sequence(
        self, node: yaml.SequenceNode, deep: bool = False
    ) -> tuple[typing.Any, ...]:
        return tuple(super().construct_sequence(node, deep))


_YAMLFrozenLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    _YAMLFrozenLoader._construct_sequence,
)


def _glob_include_patterns(patterns: str | Sequence[str], root_dir: Path) -> list[str]:
    """expand ``!include`` glob patterns into paths, sorted within each pattern.

    Args:
        patterns: one glob pattern, or a sequence of patterns
        root_dir: the directory under which relative patterns are globbed; an
            absolute pattern is globbed under its own parent directory

    Raises:
        FileNotFoundError: when a pattern matches no file
    """

    def glob_path(s: str) -> list[str]:
        p = Path(s)
        if p.is_absolute():
            rel = p.parent
            s = p.name
        else:
            rel = root_dir
        paths = [str(g) for g in rel.glob(s)]
        if len(paths) == 0:
            raise FileNotFoundError(s)
        return sorted(paths)

    if isinstance(patterns, str):
        return glob_path(patterns)

    values: list[str] = []
    for pattern in patterns:
        values.extend(glob_path(pattern))
    return values


def _expand_paths(node: yaml.Node, root_dir: Path) -> list[str]:
    """expand the ``!include`` patterns in `node` into sorted paths.

    Args:
        node: a scalar node with one glob pattern, or a sequence node of patterns
        root_dir: the directory under which relative patterns are globbed

    Raises:
        FileNotFoundError: when a pattern matches no file
        TypeError: when `node` is neither a scalar nor a sequence
    """

    if isinstance(node.value, str):
        patterns: str | list[str] = node.value
    elif isinstance(node.value, (list, tuple)):
        patterns = [n.value for n in node.value]
    else:
        raise TypeError(f'invalid tag type {type(node.value)!r} in !import')

    return _glob_include_patterns(patterns, root_dir)


class _YAMLIncludeConstructor(yaml.Loader):
    _lock = threading.RLock()

    def __init__(self, path: str | Path) -> None:
        self.nested_paths: list[Path] = [Path(path)]

    def __enter__(self) -> None:
        self._lock.acquire()
        yaml.add_constructor(_INCLUDE_TAG, self, Loader=_YAMLFrozenLoader)

    def __exit__(self, *args: object) -> None:
        self._lock.release()

    def get_include_path(self, s: str) -> Path:
        """push and return the path of an included file.

        A relative `s` is kept relative to the working directory, not to the
        including file.
        """
        p = Path(s)
        self.nested_paths.append(p)
        return p

    def pop_include_path(self) -> None:
        self.nested_paths.pop()

    def __call__(self, loader: _YAMLFrozenLoader, node: yaml.Node) -> typing.Any:
        if not node.tag.startswith(_INCLUDE_TAG):
            raise ValueError(f'unknown tag {node.tag!r}')

        values = _expand_paths(node, root_dir=self.nested_paths[-1].parent)

        content = []
        for v in values:
            path = self.get_include_path(v)
            with open(path, 'rb') as stream:
                content.append(yaml.load(stream, _YAMLFrozenLoader))
            self.pop_include_path()

        return merge_included(content)


_S = typing.TypeVar('_S', bound=specs.SpecBase)


@typing.overload
def decode_from_yaml_file(
    path: str | Path, *, type: type[dict] = ...
) -> dict[str, typing.Any]: ...


@typing.overload
def decode_from_yaml_file(path: str | Path, *, type: type[_S]) -> _S: ...


def decode_from_yaml_file(
    path: str | Path, *, type: type[specs.SpecBase | dict] = dict
) -> dict[str, typing.Any] | specs.SpecBase:
    """decode a YAML file into a dict or a spec struct.

    An ``!include`` tag is expanded relative to the directory of the file that
    contains the tag. YAML sequences decode as tuples.

    Args:
        path: the YAML file
        type: `dict` for the mapping as decoded, or a `specs.SpecBase` subclass to
            convert the mapping into

    Returns:
        the decoded dict, or an instance of `type`

    Raises:
        TypeError: when `type` is neither `dict` nor a `specs.SpecBase` subclass
        FileNotFoundError: when `path` or an ``!include`` pattern matches no file
        yaml.YAMLError: when the document is malformed
        msgspec.ValidationError: when the document does not match `type`
    """

    with open(path) as f, _YAMLIncludeConstructor(path):
        obj = yaml.load(f, _YAMLFrozenLoader)

    if issubclass(type, dict):
        return obj
    elif issubclass(type, specs.SpecBase):
        return type.from_dict(obj)
    else:
        raise TypeError(f'unsupported type {type!r}')


def yaml_flow_text(value: typing.Any) -> str:
    """return `value` as one line of YAML flow text.

    A string that YAML 1.1 would read as another type is quoted, so the text reads
    back as the same value; `None` is written as ``null``.
    """
    text = yaml.safe_dump(value, default_flow_style=True, width=float('inf'))
    return text.strip().removesuffix('\n...').strip()


# %% JSON pointers (RFC 6901)


def escape_pointer_token(token: str) -> str:
    """return `token` with ``~`` written as ``~0`` and ``/`` as ``~1``"""
    return token.replace('~', '~0').replace('/', '~1')


def unescape_pointer_token(token: str) -> str:
    """return `token` with ``~1`` read as ``/`` and ``~0`` as ``~``"""
    return token.replace('~1', '/').replace('~0', '~')


def json_pointer(parts: Sequence[typing.Any]) -> str:
    """return the JSON pointer of the path `parts`, one escaped `str` token per part.

    An empty `parts` gives the empty pointer, which names the whole document.
    """
    return ''.join(f'/{escape_pointer_token(str(part))}' for part in parts)


# %% provenance


@dataclasses.dataclass
class YAMLSource:
    """one YAML file read by `load_yaml_provenance`, as a round-trip document"""

    path: Path
    """absolute path of the file"""

    doc: typing.Any
    """the round-trip document (`CommentedMap`, `CommentedSeq`, or a scalar), which
    keeps comments, anchors, quotes, key order, and tags"""


@dataclasses.dataclass
class ProvenanceTree:
    """the merged document of a YAML spec file and the file position of every node.

    JSON pointers (RFC 6901) address nodes in `tree`; a pointer token is the `str`
    of the mapping key or the sequence index. `origin` maps each key and element
    of `tree` to the file that holds its value and the pointer within that
    file's document; `regions` maps each node produced by an ``!include`` tag to
    the files merged there, in merge order.
    """

    root: Path
    """absolute path of the top file"""

    tree: typing.Any
    """the merged document as `dict`, `list`, and scalars, equal to the
    `decode_from_yaml_file` result with tuples read as lists"""

    files: dict[Path, YAMLSource]
    """every file read, keyed by absolute path, in first-read order"""

    origin: dict[str, tuple[Path, str]]
    """JSON pointer in `tree` to the file that holds the value and the JSON
    pointer within that file's document; a key set by several files of a merged
    include maps to the last of them"""

    regions: dict[str, tuple[Path, ...]]
    """JSON pointer of each node that an ``!include`` tag produced to the files
    merged there, in merge order"""


@dataclasses.dataclass(frozen=True)
class YAMLEdit:
    """one change to the merged document of a `ProvenanceTree`"""

    pointer: str
    """JSON pointer (RFC 6901) in the merged tree; ``insert`` takes the pointer
    of the sequence, with or without a trailing ``/-``"""

    op: EditOp
    """``'set'``, ``'delete'``, or ``'insert'``"""

    value_text: typing.Optional[str] = None
    """YAML text of the new value for ``set`` and ``insert``; `None` for ``delete``"""


@dataclasses.dataclass
class _FileLoad:
    """the converted document of one file, with pointers relative to the file root"""

    tree: typing.Any
    origin: dict[str, tuple[Path, str]]
    regions: dict[str, tuple[Path, ...]]


def load_yaml_provenance(path: str | Path) -> ProvenanceTree:
    """load a YAML spec file and record the file that holds each value.

    ``!include`` tags are expanded as in `decode_from_yaml_file`: a pattern is
    globbed relative to the directory of the file that contains the tag, several
    matches of mappings are merged with later keys overriding, and several
    matches of sequences are concatenated. Merge keys (``<<``) are resolved into
    `tree` with the file of the mapping as origin. Scalars resolve under the
    PyYAML rules (YAML 1.1: ``3750e6`` is a string, ``yes`` is a bool). When a
    key repeats in one mapping, the last value is kept.

    Args:
        path: the YAML file

    Returns:
        the merged tree with the origin of every key and element, and the
        round-trip document of every file read, for `apply_yaml_edits`

    Raises:
        FileNotFoundError: when `path` or an ``!include`` pattern matches no file
        ValueError: when a tag other than ``!include`` is present
        TypeError: when the files of one ``!include`` mix mappings and sequences,
            or an ``!include`` tag is on a mapping
    """
    rt = _round_trip_yaml()
    files: dict[Path, YAMLSource] = {}
    root = Path(path).resolve()
    loaded = _load_yaml_source(root, files, rt)
    return ProvenanceTree(
        root=root,
        tree=loaded.tree,
        files=files,
        origin=loaded.origin,
        regions=loaded.regions,
    )


def apply_yaml_edits(
    prov: ProvenanceTree, edits: Sequence[YAMLEdit]
) -> tuple[Path, ...]:
    """write edits to the merged document back into the files that hold the values.

    Each edit is applied to the round-trip document of the file named by
    `prov.origin` for its pointer, and the touched files are rewritten with
    comments, anchors, quotes, key order, and untouched ``!include`` tags kept.
    A value is parsed from `value_text` as YAML, so ``3750e6`` is written as
    ``3750e6`` and ``[0, 1]`` as a flow sequence. A ``set`` on a key that does
    not exist appends the key to the mapping of the parent pointer; when the
    parent is a merged ``!include``, the last file of the region receives the
    key. An ``insert`` appends to the sequence at the pointer, into the last
    file of the region when the sequence came from a merge. A ``delete`` removes
    the key or element. Each file is replaced atomically by `os.replace` of a
    temporary file in the same directory. `prov` is not updated, so the caller
    reloads with `load_yaml_provenance` before further edits.

    Args:
        prov: the tree returned by `load_yaml_provenance`
        edits: the edits, applied in order

    Returns:
        the paths written, in sorted order

    Raises:
        KeyError: when a pointer resolves to no node, or ``delete`` names a key
            that a merge key (``<<``) supplies rather than the mapping itself
        ValueError: for an `op` other than ``'set'``, ``'delete'``, or
            ``'insert'``, or a ``set``/``insert`` without `value_text`
        TypeError: when ``insert`` targets a node that is not a sequence
    """
    for edit in edits:
        if edit.op not in _EDIT_OPS:
            raise ValueError(f'unknown op {edit.op!r}; expected one of {_EDIT_OPS}')
        if edit.op != 'delete' and edit.value_text is None:
            raise ValueError(f'value_text is required for op {edit.op!r}')

    rt = _round_trip_yaml()
    touched = {_apply_yaml_edit(prov, edit, rt) for edit in edits}

    written = tuple(sorted(touched))
    for path in written:
        _write_yaml_atomic(prov.files[path].doc, path)
    return written


def tree_edits(
    current: typing.Any, target: typing.Any, pointer: str = ''
) -> list[YAMLEdit]:
    """return the edits that turn `current` into `target` at `pointer`.

    Mappings are compared key by key and lists element by element when the lengths
    agree, so an unchanged value is left in place; a list with a changed length and
    every other difference is replaced as a whole. A value is written through
    `yaml_flow_text`.

    Args:
        current: the merged tree, or the node of the tree at `pointer`
        target: the tree or node to reach
        pointer: the JSON pointer of `current` in the merged tree

    Returns:
        the edits in application order: deletions of the keys absent from
        `target`, then sets
    """
    if isinstance(current, dict) and isinstance(target, dict):
        edits = [
            YAMLEdit(pointer + json_pointer([key]), 'delete', None)
            for key in current
            if key not in target
        ]
        for key, value in target.items():
            child = pointer + json_pointer([key])
            if key not in current:
                edits.append(YAMLEdit(child, 'set', yaml_flow_text(value)))
            else:
                edits.extend(tree_edits(current[key], value, child))
        return edits
    if (
        isinstance(current, list)
        and isinstance(target, list)
        and len(current) == len(target)
    ):
        edits = []
        for i, (a, b) in enumerate(zip(current, target)):
            edits.extend(tree_edits(a, b, pointer + json_pointer([i])))
        return edits
    if current == target and type(current) is type(target):
        return []
    return [YAMLEdit(pointer, 'set', yaml_flow_text(target))]


def _load_yaml_source(
    path: Path, files: dict[Path, YAMLSource], rt: ruamel.yaml.YAML
) -> _FileLoad:
    """load `path` once into `files` and convert its document into a `_FileLoad`"""
    if path not in files:
        with open(path, encoding='utf-8') as f:
            files[path] = YAMLSource(path=path, doc=rt.load(f))
    source = files[path]
    load = _FileLoad(tree=None, origin={}, regions={})
    load.tree = _convert_node(source.doc, '', source, load, files, rt)
    return load


def _convert_node(
    node: typing.Any,
    pointer: str,
    source: YAMLSource,
    load: _FileLoad,
    files: dict[Path, YAMLSource],
    rt: ruamel.yaml.YAML,
) -> typing.Any:
    """convert a round-trip node into plain Python, recording origins into `load`"""
    comments = ruamel.yaml.comments

    if isinstance(node, comments.TaggedScalar) or (
        isinstance(node, (comments.CommentedMap, comments.CommentedSeq))
        and node.tag.value is not None
    ):
        return _convert_include(node, pointer, source, load, files, rt)

    if isinstance(node, comments.CommentedMap):
        mapping: dict[typing.Any, typing.Any] = {}
        for key, value in node.items():
            child = f'{pointer}/{escape_pointer_token(_key_token(key))}'
            load.origin[child] = (source.path, child)
            mapping[_plain_key(key)] = _convert_node(
                value, child, source, load, files, rt
            )
        return mapping

    if isinstance(node, comments.CommentedSeq):
        items: list[typing.Any] = []
        for i, value in enumerate(node):
            child = f'{pointer}/{i}'
            load.origin[child] = (source.path, child)
            items.append(_convert_node(value, child, source, load, files, rt))
        return items

    return _plain_scalar(node)


def _convert_include(
    node: typing.Any,
    pointer: str,
    source: YAMLSource,
    load: _FileLoad,
    files: dict[Path, YAMLSource],
    rt: ruamel.yaml.YAML,
) -> typing.Any:
    """expand an ``!include`` node and merge the files into `load` at `pointer`"""
    comments = ruamel.yaml.comments

    tag = node.tag.value
    if tag != _INCLUDE_TAG:
        raise ValueError(f'unknown tag {tag!r}')

    if isinstance(node, comments.TaggedScalar):
        patterns: str | list[str] = str(node.value)
    elif isinstance(node, comments.CommentedSeq):
        patterns = [str(v) for v in node]
    else:
        raise TypeError(f'invalid tag type {type(node)!r} in {_INCLUDE_TAG}')

    paths = [
        Path(p).resolve() for p in _glob_include_patterns(patterns, source.path.parent)
    ]
    loads = [_load_yaml_source(p, files, rt) for p in paths]
    result = merge_included([sub.tree for sub in loads])

    if len(loads) == 1:
        _adopt_subtree(load, loads[0], '', pointer)
    elif isinstance(result, dict):
        for sub in loads:
            for key in sub.tree:
                token = escape_pointer_token(_key_token(key))
                _adopt_subtree(load, sub, f'/{token}', f'{pointer}/{token}')
    else:
        offset = 0
        for sub in loads:
            for i in range(len(sub.tree)):
                _adopt_subtree(load, sub, f'/{i}', f'{pointer}/{offset + i}')
            offset += len(sub.tree)

    load.regions[pointer] = tuple(paths)
    return result


def _adopt_subtree(load: _FileLoad, sub: _FileLoad, old: str, new: str) -> None:
    """move the origins and regions of `sub` under `old` into `load` under `new`,
    replacing the entries of an earlier file below `new`"""
    for entries in (load.origin, load.regions):
        for key in [k for k in entries if k.startswith(new + '/')]:
            del entries[key]
    load.origin.update(_rebase_pointers(sub.origin, old, new))
    load.regions.update(_rebase_pointers(sub.regions, old, new))


_T = typing.TypeVar('_T')


def _rebase_pointers(entries: dict[str, _T], old: str, new: str) -> dict[str, _T]:
    """return the entries at or below pointer `old` with the prefix replaced by `new`"""
    return {
        new + k[len(old) :]: v
        for k, v in entries.items()
        if k == old or k.startswith(old + '/')
    }


def _plain_scalar(value: typing.Any) -> typing.Any:
    """return the builtin equivalent of a round-trip scalar"""
    if isinstance(value, ruamel.yaml.scalarbool.ScalarBoolean):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return value


def _plain_key(key: typing.Any) -> typing.Any:
    """return a mapping key as a builtin, with a sequence key as a tuple"""
    if isinstance(key, ruamel.yaml.comments.CommentedKeySeq):
        return tuple(_plain_key(k) for k in key)
    return _plain_scalar(key)


def _key_token(key: typing.Any) -> str:
    """return the JSON pointer token of a mapping key, before escaping"""
    if isinstance(key, ruamel.yaml.comments.CommentedKeySeq):
        return str(tuple(_plain_key(k) for k in key))
    return str(_plain_scalar(key))


def _split_pointer(pointer: str) -> tuple[str, str]:
    """return the parent pointer and the unescaped last token of `pointer`.

    Raises:
        KeyError: when `pointer` is empty or does not start with ``/``
    """
    if not pointer.startswith('/'):
        raise KeyError(pointer)
    parent, _, token = pointer.rpartition('/')
    return parent, unescape_pointer_token(token)


def _mapping_key(node: typing.Any, token: str) -> typing.Any:
    """return the key of the round-trip mapping `node` with pointer token `token`.

    Raises:
        KeyError: when no key of `node` has that token
    """
    for key in node:
        if _key_token(key) == token:
            return key
    raise KeyError(token)


def _child_node(node: typing.Any, token: str) -> typing.Any:
    """return the child of a round-trip mapping or sequence at `token`.

    Raises:
        KeyError: when `node` is a scalar or has no child at `token`
    """
    if isinstance(node, ruamel.yaml.comments.CommentedSeq):
        try:
            return node[int(token)]
        except (ValueError, IndexError):
            raise KeyError(token) from None
    if isinstance(node, ruamel.yaml.comments.CommentedMap):
        return node[_mapping_key(node, token)]
    raise KeyError(token)


def _resolve_pointer(doc: typing.Any, pointer: str) -> typing.Any:
    """return the node of the round-trip document `doc` at the JSON pointer"""
    node = doc
    for token in pointer.split('/')[1:]:
        node = _child_node(node, unescape_pointer_token(token))
    return node


def _locate_container(prov: ProvenanceTree, pointer: str) -> tuple[Path, typing.Any]:
    """return the file and round-trip node that receive a new child of `pointer`.

    A node produced by a merged ``!include`` resolves to the root of the last
    file of the region.

    Raises:
        KeyError: when `pointer` names no node of `prov.tree`
    """
    if pointer in prov.regions:
        path = prov.regions[pointer][-1]
        return path, prov.files[path].doc
    if pointer == '':
        return prov.root, prov.files[prov.root].doc
    if pointer in prov.origin:
        path, local = prov.origin[pointer]
        return path, _resolve_pointer(prov.files[path].doc, local)
    raise KeyError(pointer)


def _locate_existing(
    prov: ProvenanceTree, pointer: str
) -> tuple[Path, typing.Any, str]:
    """return the file, round-trip parent node, and token of the node at `pointer`.

    Raises:
        KeyError: when `pointer` names no key or element of `prov.tree`
    """
    try:
        path, local = prov.origin[pointer]
    except KeyError:
        raise KeyError(pointer) from None
    parent_local, token = _split_pointer(local)
    return path, _resolve_pointer(prov.files[path].doc, parent_local), token


def _apply_yaml_edit(
    prov: ProvenanceTree, edit: YAMLEdit, rt: ruamel.yaml.YAML
) -> Path:
    """apply one edit to the round-trip documents of `prov` and return its file"""
    comments = ruamel.yaml.comments

    if edit.op == 'delete':
        path, parent, token = _locate_existing(prov, edit.pointer)
        if isinstance(parent, comments.CommentedSeq):
            del parent[int(token)]
        else:
            key = _mapping_key(parent, token)
            literal_keys = [k for k, _ in parent.non_merged_items()]
            if not any(k == key for k in literal_keys):
                raise KeyError(
                    f'{edit.pointer!r} is supplied by a merge key (<<) of '
                    f'{path}, not by the mapping itself'
                )
            del parent[key]
        return path

    value = rt.load(edit.value_text)

    if edit.op == 'set':
        if edit.pointer in prov.origin:
            path, parent, token = _locate_existing(prov, edit.pointer)
            if isinstance(parent, comments.CommentedSeq):
                parent[int(token)] = value
            else:
                parent[_mapping_key(parent, token)] = value
            return path

        parent_pointer, token = _split_pointer(edit.pointer)
        path, parent = _locate_container(prov, parent_pointer)
        if not isinstance(parent, comments.CommentedMap):
            raise KeyError(edit.pointer)
        parent[token] = value
        return path

    pointer = edit.pointer.removesuffix('/-')
    path, node = _locate_container(prov, pointer)
    if not isinstance(node, comments.CommentedSeq):
        raise TypeError(f'{pointer!r} is not a sequence')
    node.append(value)
    return path


def _write_yaml_atomic(doc: typing.Any, path: Path) -> None:
    """dump the round-trip `doc` to a temporary file beside `path`, then replace `path`"""
    rt = _round_trip_yaml(indent=_detect_indent(doc))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            rt.dump(doc, f)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _child_col(node: typing.Any, key: typing.Any) -> int | None:
    """return the column of the child of the round-trip `node` at `key`"""
    # a key or element added since the load has no line and column record
    data = node.lc.data
    if data is None or key not in data:
        return None
    return data[key][1]


def _is_block(value: typing.Any, cls: type) -> bool:
    """return whether `value` is a `cls` node in block (not flow) style"""
    return isinstance(value, cls) and not value.fa.flow_style()


def _detect_indent(doc: typing.Any) -> tuple[int, int, int]:
    """return the (mapping, sequence, offset) indents of `doc` from its node columns.

    The first block mapping nested in a mapping sets the mapping indent, and the
    first block sequence sets the sequence indent and dash offset; a style
    absent from `doc` takes the default (2, 4, 2).
    """
    comments = ruamel.yaml.comments
    mapping: int | None = None
    sequence: int | None = None
    offset: int | None = None

    def walk(node: typing.Any, col: int) -> None:
        nonlocal mapping, sequence, offset
        if isinstance(node, comments.CommentedMap):
            for key, value in node.non_merged_items():
                key_col = _child_col(node, key)
                if key_col is None:
                    continue
                if mapping is None and _is_block(value, comments.CommentedMap):
                    mapping = value.lc.col - key_col
                walk(value, key_col)
        elif isinstance(node, comments.CommentedSeq):
            if offset is None and _is_block(node, comments.CommentedSeq):
                offset = node.lc.col - col
                item_col = _child_col(node, 0)
                if item_col is not None:
                    sequence = item_col - col
            for value in node:
                walk(value, node.lc.col)

    walk(doc, 0)
    return (
        2 if mapping is None or mapping < 1 else mapping,
        4 if sequence is None or sequence < 2 else sequence,
        2 if offset is None or offset < 0 else offset,
    )


@functools.cache
def _round_trip_classes() -> tuple[type, type, type]:
    """return the resolver, constructor, and representer classes of the round-trip loader"""
    # PyYAML resolves with its YAML 1.1 tables, under which 3750e6 is a string;
    # ruamel resolves 1.1 floats without a mantissa dot, so the tables are
    # copied from PyYAML to keep the merged tree equal to decode_from_yaml_file
    implicit = {
        first: list(resolvers)
        for first, resolvers in yaml.resolver.Resolver.yaml_implicit_resolvers.items()
    }

    class Resolver(ruamel.yaml.resolver.VersionedResolver):
        def __init__(
            self,
            version: typing.Any = None,
            loader: typing.Any = None,
            loadumper: typing.Any = None,
        ) -> None:
            super().__init__((1, 1), loader, loadumper)

        @property
        def versioned_resolver(self) -> dict[typing.Any, list[typing.Any]]:
            return implicit

    class Constructor(ruamel.yaml.constructor.RoundTripConstructor):
        def check_mapping_key(
            self,
            node: typing.Any,
            key_node: typing.Any,
            mapping: typing.Any,
            key: typing.Any,
            value: typing.Any,
        ) -> bool:
            # PyYAML keeps the last value of a repeated key
            return True

    class Representer(ruamel.yaml.representer.RoundTripRepresenter):
        pass

    Representer.add_representer(
        type(None),
        lambda representer, data: representer.represent_scalar(
            'tag:yaml.org,2002:null', 'null'
        ),
    )

    return Resolver, Constructor, Representer


def _round_trip_yaml(indent: tuple[int, int, int] = (2, 4, 2)) -> ruamel.yaml.YAML:
    """return a round-trip `YAML` that resolves scalars as PyYAML does.

    Args:
        indent: the (mapping, sequence, offset) indents used on dump
    """
    Resolver, Constructor, Representer = _round_trip_classes()
    rt = ruamel.yaml.YAML(typ='rt', pure=True)
    rt.Resolver = Resolver
    rt.Constructor = Constructor
    rt.Representer = Representer
    rt.preserve_quotes = True
    rt.width = 4096
    mapping, sequence, offset = indent
    rt.indent(mapping=mapping, sequence=sequence, offset=offset)
    return rt
