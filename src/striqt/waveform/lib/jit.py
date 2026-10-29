"""JIT implementations of numerical functions specific to CPU and CUDA"""

from __future__ import annotations as __

import hashlib
import os

# must be set before numba is first imported; cp_cpu_jit.py and cp_cuda_jit.py import
# this module before numba
if not os.environ.get('NUMBA_CACHE_DIR'):
    try:
        from .util import cache_dir

        _dir = cache_dir() / 'numba'
        _dir.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(_dir)
    except OSError:
        pass

# numba invalidates .nbc cache files when the source file mtime changes, which happens on
# every pip reinstall even when the content is identical; a content hash replaces the
# mtime in the stamp
from numba.core.caching import _SourceFileBackedLocatorMixin as _Locator


def _stamp_by_content(self) -> tuple[str, int] | tuple[float, int]:
    try:
        with open(self._py_file, 'rb') as fd:
            data = fd.read()
        return (hashlib.sha1(data).hexdigest(), len(data))
    except OSError:
        st = os.stat(self._py_file)
        return (st.st_mtime, st.st_size)


_Locator.get_source_stamp = _stamp_by_content
