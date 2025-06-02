from . import base
from .. import specs, util

import pickle
import typing

import rpyc

if typing.TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    import labbench as lb
else:
    pd = util.lazy_import('pandas')
    np = util.lazy_import('numpy')
    lb = util.lazy_import('labbench')


class RadioProcessService(rpyc.Service):
    exposed_namespace = {}
    conn = None

    def on_connect(self, conn: rpyc.Service):
        if self.conn is None:
            self.conn = conn
        else:
            raise ValueError('already connected')

    def on_disconnect(self, conn: rpyc.Service):
        if conn is self.conn:
            self.conn = None

    def __init__(self, radio_cls_name: str):
        self.radio = base.find_radio_cls_by_name(radio_cls_name)

    # TODO: wrap self.radio._read_stream to pull from a queue?
    # TODO: implement a low-level acquisition loop?

    def exposed_acquire(
        self,
        capture: specs.RadioCapture,
        next_capture: typing.Union[specs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, 'pd.Timestamp']:
        iq = self.radio.acquire(capture, next_capture, correction)

        if self.conn is None:
            raise RuntimeError('not connected')

        return self.conn.root.remote_shared_array(iq)

    def exposed_read_stream(self, samples: int):
        iq = self.radio._read_stream(samples)
        return self.conn.root.remote_shared_array(iq)

    def exposed_setup(self, radio_setup: specs.RadioSetup, analysis=None):
        radio_setup = rpyc.utils.classic.obtain(radio_setup)
        return self.radio.setup(radio_setup, analysis)

    def exposed_arm(self, capture: specs.RadioCapture):
        capture = rpyc.utils.classic.obtain(capture)
        return self.radio.arm(capture)

    def exposed_get_capture_struct(self, cls=specs.RadioCapture) -> specs.RadioCapture:
        return self.radio.get_capture_struct(cls)


class SharedMemoryClientService(rpyc.Service):
    exposed_namespace = {}
    conn = None

    def on_connect(self, conn: rpyc.Service):
        if self.conn is None:
            self.conn = conn
        else:
            raise ValueError('already connected')

    def on_disconnect(self, conn: rpyc.Service):
        if conn is self.conn:
            self.conn = None

    def exposed_reference_shared_array(
        self, shared_array: util.NDSharedArray, free_on_del=False
    ):
        s = pickle.dumps(shared_array)
        return util.NDSharedArray.from_pickle(s, free_on_del)


class RadioProcessClient(rpyc.Client):
    pass


def connect(host='localhost', port=4567) -> rpyc.Connection:
    """connect to a remote sensor sensor.

    The returned connection object contains a `root` attribute that
    exposes remote wrappers for sweep iterators.

    Example::

        remote = connect_to_sensor('localhost')
        remote.root.iter_sweep()
    """

    # support host:port
    host, *extra = host.split(',', 1)
    if len(extra) != 0:
        port = int(extra[0])

    rpyc.utils.factory.connect_multiprocess()

    return rpyc.connect_subproc(
        host=host,
        port=port,
        config={'logger': lb.logger, 'allow_pickle': True},
        service=_ClientService,
    )
