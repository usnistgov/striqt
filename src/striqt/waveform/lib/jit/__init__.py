"""JIT implementations of numerical functions specific to CPU and CUDA"""

import hashlib
import os
import pathlib

# must be set before numba is first imported; this module loads before cpu.py or cuda.py
if not os.environ.get('NUMBA_CACHE_DIR'):
    try:
        from platformdirs import user_cache_dir

        _dir = pathlib.Path(user_cache_dir('striqt')) / 'numba'
        _dir.mkdir(parents=True, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = str(_dir)
    except OSError:
        pass

# numba invalidates .nbc cache files when the source file mtime changes, which happens on
# every pip reinstall even when the content is identical — use content hash instead
from numba.core.caching import _SourceFileBackedLocatorMixin as _Locator


def _stamp_by_content(self):
    try:
        data = open(self._py_file, 'rb').read()
        return (hashlib.sha1(data).hexdigest(), len(data))
    except OSError:
        st = os.stat(self._py_file)
        return (st.st_mtime, st.st_size)


_Locator.get_source_stamp = _stamp_by_content
