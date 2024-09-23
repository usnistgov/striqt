from __future__ import annotations
import enum
import functools
import numbers
import psutil
import socket
import typing
import uuid

from pathlib import Path
import dulwich

from channel_analysis import type_stubs
from . import util

if typing.TYPE_CHECKING:
    import labbench as lb
    import xarray as xr
    import dulwich.repo
    import dulwich.porcelain
else:
    lb = util.lazy_import('labbench')
    xr = util.lazy_import('xarray')
    dulwich.repo = util.lazy_import('dulwich.repo')
    dulwich.porcelain = util.lazy_import('dulwich.porcelain')


METADATA_VERSION = '0.0'


@functools.lru_cache(8)
def _find_repo_in_parents(path: Path) -> 'dulwich.repo.Repo':
    """find a git repository in path, or in the first parent to contain one"""
    path = Path(path).absolute()

    try:
        return dulwich.repo.Repo(str(path))
    except dulwich.repo.NotGitRepository as ex:
        if not path.is_dir() or path.parent is path:
            raise

        try:
            return _find_repo_in_parents(path.parent)
        except dulwich.repo.NotGitRepository:
            ex.args = ex.args[0] + ' (and parent directories)'
            raise ex


def _filter_types(obj: dict | list | tuple):
    if isinstance(obj, dict):
        return {k: _filter_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_filter_types(v) for v in obj]
    elif isinstance(obj, enum.Enum) or obj is None:
        return None
    elif isinstance(obj, (numbers.Number, bool, str)):
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


@functools.lru_cache(8)
def _compute_status_meta(keys: tuple):
    return {
        'coords': type_stubs.CoordinatesType({'compute_status_category': list(keys)}),
        'attrs': {
            'hostname': socket.gethostname(),
            'disk_size': _psutil_to_dict('disk_usage', '.')['total'],
            'swap_size': _psutil_to_dict('swap_memory')['total'],
            'mem_size': _psutil_to_dict('virtual_memory')['total'],
        },
    }


def package_log_messages(host: 'lb.Host') -> dict[str, xr.DataArray]:
    """package logger messages from labbench._host.Host.log into xarray DataArrays"""

    messages = host.log

    fields = list(messages[0].keys())
    flat = [list(m.values()) for m in messages]

    coords = {
        'message_index': range(len(flat)),
        'message_field': fields,
    }

    array = (
        xr.DataArray(flat, coords=coords, name='messages')
        .drop_sel({'message_field': ('thread', 'object_log_name')})
        .astype('str')
    )

    return {'host_log': array}


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

    names = dulwich.porcelain.status(repo, untracked_files='no').unstaged
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


@functools.lru_cache(8)
def _temperature_coords(keys: tuple):
    return xr.Coordinates({'temperature_sensor': list(keys)})


@functools.lru_cache(8)
def _temperature_coords(keys: tuple):
    return xr.Coordinates({'temperature_sensor': list(keys)})


def package_host_resources(host: Host) -> dict[str, xr.DataArray]:
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

    return {'host_temperature': temperature, 'host_data_usage': compute_status}
