from __future__ import annotations
from numbers import Number
from enum import Enum
import psutil
import xarray as xr
from functools import lru_cache

from dulwich.repo import Repo, NotGitRepository
from dulwich import porcelain
from pathlib import Path
import socket
import uuid

METADATA_VERSION = '0.0'


@lru_cache(8)
def _find_repo_in_parents(path: Path) -> Repo:
    """find a git repository in path, or in the first parent to contain one"""
    path = Path(path).absolute()

    try:
        return Repo(str(path))
    except NotGitRepository as ex:
        if not path.is_dir() or path.parent is path:
            raise

        try:
            return _find_repo_in_parents(path.parent)
        except NotGitRepository:
            ex.args = ex.args[0] + ' (and parent directories)'
            raise ex


def _filter_types(obj: dict | list | tuple):
    if isinstance(obj, dict):
        return {k: _filter_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_filter_types(v) for v in obj]
    elif isinstance(obj, Enum) or obj is None:
        return None
    elif isinstance(obj, (Number, bool, str)):
        return obj
    else:
        raise TypeError(f'object "{repr(obj)}" has unsupported type')


def _asdict(obj):
    if hasattr(obj, '_asdict'):
        ret = obj._asdict()
    elif isinstance(obj, list):
        ret = dict(enumerate(obj))
    else:
        ret = dict(obj)

    return _filter_types(ret)


def _psutil_to_dict(name, *args, **kws):
    try:
        func = getattr(psutil, name)
    except AttributeError:
        return None

    ret = func(*args, **kws)

    return _asdict(ret)


@lru_cache(8)
def _compute_status_meta(keys: tuple):
    return {
        'coords': xr.Coordinates({'compute_status_category': list(keys)}),
        'attrs': {
            'hostname': socket.gethostname(),
            'disk_size': _psutil_to_dict('disk_usage', '.')['total'],
            'swap_size': _psutil_to_dict('swap_memory')['total'],
            'mem_size': _psutil_to_dict('virtual_memory')['total'],
        },
    }


def git_unstaged_changes(repo_or_path='.') -> list[str]:
    """returns a list of files in a git repository with unstaged changes.

    Args:
        repo_or_path: root or child path to locate the repository
    """
    try:
        repo = _find_repo_in_parents(repo_or_path)
    except NotGitRepository:
        return []

    if repo is None:
        return []

    names = porcelain.status(repo, untracked_files='no').unstaged
    return [n.decode() for n in names]


def host_metadata(search_path='.'):
    try:
        repo = _find_repo_in_parents(search_path)
    except NotGitRepository:
        return {}

    if repo is None:
        return {}

    repo_info = {
        'git_remote': repo.get_config().get(('remote', 'origin'), 'url').decode(),
        'git_commit': repo.head().decode(),
        # 'git_unstaged_changes': git_unstaged_changes('.')
        'host_uuid': hex(uuid.getnode()),
        'host_name': socket.gethostname(),
    }

    return repo_info


@lru_cache(8)
def _temperature_coords(keys: tuple):
    return xr.Coordinates({'temperature_sensor': list(keys)})


@lru_cache(8)
def _temperature_coords(keys: tuple):
    return xr.Coordinates({'temperature_sensor': list(keys)})


def host_index_variables(temperature={}):
    compute_status = {
        'disk_usage_percentage': _psutil_to_dict('disk_usage', '.')['percent'],
        'swap_usage_percentage': _psutil_to_dict('swap_memory')['percent'],
        'mem_usage_percentage': _psutil_to_dict('virtual_memory')['percent'],
    }

    compute_status = xr.DataArray(
        list(compute_status.values()),
        **_compute_status_meta(tuple(compute_status.keys())),
    )

    temperature = _psutil_to_dict('sensors_temperatures') or {}
    temperature = {k: v[0][1] for k, v in temperature.items()}

    temperature = xr.DataArray(
        list(temperature.values()),
        coords=_temperature_coords(tuple(temperature.keys())),
        attrs={'units': 'C'},
    )

    return {'temperature': temperature, 'compute_status': compute_status}
