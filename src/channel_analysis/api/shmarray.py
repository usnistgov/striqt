"""shared memory arrays"""

import functools
import numpy as np
import pickle
from multiprocessing import shared_memory


class NDSharedArray(np.ndarray):
    """an ndarray subclass for inter-process communication that serializes to shared memory
    reference information instead of raw values.

    Notes:
    * If pickled, the shared memory backing will be freed _only if the object is deserialized and deleted_
    * Processes that deserialize this object must share memory with the process that creates the NDSharedArray
    """

    # References:
    #   https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    shm: 'shared_memory.SharedMemory'
    _free_on_del: bool = True

    def __new__(cls, input_array, shm: 'shared_memory.SharedMemory' = None):
        if shm is not None:
            obj = np.asarray(input_array).view(cls)
            obj.shm = shm
        elif isinstance(input_array, NDSharedArray):
            obj = input_array.view(cls)
            obj.shm = input_array.shm
        else:
            if not isinstance(input_array, np.ndarray):
                input_array = np.asanyarray(input_array)
            obj = empty_shared_array(input_array.shape, dtype=input_array.dtype)
            obj[:] = input_array.view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.shm = getattr(obj, 'shm', None)

    def __del__(self):
        if self.shm is None:
            return
        elif self._free_on_del:
            try:
                self.free()
            except ImportError:
                # arises on exceptions during python shutdown
                pass
        else:
            self.shm.close()

    def persist(self):
        self._free_on_del = False

    def free(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass

    def __reduce__(self):
        # return the shared memory reference info for pickling
        self._free_on_del = False
        shm_info = (self.shm.name, self.shape, self.dtype.str)
        return (reference_shared_array, shm_info)

    @classmethod
    def from_pickle(cls, s: str):
        """instantiate a new NDSharedArray referenced from a pickle.

        if last_access is True, the underlying shared memory will be freed
        when this array instance is deleted.
        """
        return pickle.loads(s)


@functools.lru_cache
def _array_memory_size(shape, dtype):
    return np.prod(shape) * np.dtype(dtype).itemsize


def empty_shared_array(shape, dtype='float32'):
    shm = shared_memory.SharedMemory(size=_array_memory_size(shape, dtype), create=True)
    a = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    return NDSharedArray(a, shm)


def reference_shared_array(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    a = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
    return NDSharedArray(a, shm)
