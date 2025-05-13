from __future__ import annotations

from itertools import zip_longest
import pickle
import typing

import rpyc

from . import captures, sweeps, util
from . import specs
from .sources import find_radio_cls_by_name, is_same_resource, SourceBase

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    import labbench as lb
    import pandas as pd
    from striqt import analysis
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')
    analysis = util.lazy_import('striqt.analysis')


def _consume_warmup(controller, gen: typing.Generator[typing.Any]):
    for _ in gen:
        # avoid retaining warmup results in memory
        pass

    controller.close_radio('NullSource')


class SweepController:
    """Manage local edge sensor operation, encapsulating radio connection logic.

    This is also used by `start_sensor_server` to serve remote operations.
    """

    def __init__(self, sweep: specs.Sweep = None):
        self.radios: dict[str, SourceBase] = {}
        self.warmed_captures: set[specs.RadioCapture] = set()
        self.handlers: dict[rpyc.Connection, typing.Any] = {}

        if sweep is None:
            # wait until later
            return
        else:
            # for performance, do any warmup runs and open a device connection
            self.warmup_sweep(sweep, calibration=None)

        util.set_cuda_mem_limit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        last_ex = None

        for radio in self.radios.values():
            try:
                radio.close()
            except BaseException as ex:
                last_ex = ex

        if last_ex is not None:
            raise last_ex

    def open_radio(self, radio_setup: specs.RadioSetup):
        driver_name = radio_setup.driver
        radio_cls = find_radio_cls_by_name(driver_name)

        if driver_name in self.radios and self.radios[driver_name].isopen:
            if is_same_resource(self.radios[driver_name], radio_setup):
                lb.logger.debug(f'reusing open {repr(driver_name)}')
                return self.radios[driver_name]
            else:
                lb.logger.debug(
                    f're-opening {repr(driver_name)} to set resource={repr(radio_setup.resource)}'
                )
                self.radios[driver_name].close()
        else:
            lb.logger.debug(f'opening driver {repr(driver_name)}')

        radio = self.radios[driver_name] = radio_cls(resource=radio_setup.resource)

        if radio_setup._transient_holdoff_time is not None:
            radio._transient_holdoff_time = radio_setup._transient_holdoff_time

        radio.open()

        return radio

    def radio_id(self, driver_name: str) -> str:
        return self.radios[driver_name].id

    def close_radio(self, radio_setup: specs.RadioSetup = None):
        if radio_setup is None:
            # close all
            for name, radio in self.radios.items():
                if lb.paramattr._bases.get_class_attrs is None:
                    # accommodate a strange side effect of partially
                    # torn down python. TODO: proper fix for this
                    continue
                try:
                    radio.close()
                except BaseException as ex:
                    lb.logger.warning(f'failed to close radio {name}: {str(ex)}')
        else:
            self.radios[radio_setup.driver].close()

    def _describe_preparation(self, target_sweep: specs.Sweep) -> str:
        if (
            sweeps.sweep_touches_gpu(target_sweep)
            and target_sweep.radio_setup.warmup_sweep
        ):
            warmup_sweep = sweeps.design_warmup_sweep(
                target_sweep, skip=tuple(self.warmed_captures)
            )
        else:
            warmup_sweep = None

        msgs = []
        if target_sweep.radio_setup.driver.startswith('Null'):
            pass
        elif target_sweep.radio_setup.driver not in self.radios:
            msgs += ['opening radio']

        if warmup_sweep is not None and len(warmup_sweep.captures) > 0:
            msgs += ['preparing GPU']
        return ' and '.join(msgs)

    def warmup_sweep(self, sweep_spec: specs.Sweep, calibration, pickled=False):
        """open the radio while warming up the GPU"""

        warmup_iter = []
        warmup_sweep = None

        calls = {}

        if not sweeps.sweep_touches_gpu(sweep_spec):
            pass
        elif not sweep_spec.radio_setup.warmup_sweep:
            pass
        elif len(self.warmed_captures) == 0:
            # maybe lead to a sweep iterator
            warmup_sweep = sweeps.design_warmup_sweep(
                sweep_spec, skip=tuple(self.warmed_captures)
            )

            if len(warmup_sweep.captures) > 0:
                prep_msg = self._describe_preparation(sweep_spec)
                if prep_msg:
                    lb.logger.info(prep_msg)

                warmup_iter = self.iter_sweep(
                    warmup_sweep,
                    always_yield=True,
                    calibration=None,
                    quiet=True,
                    prepare=False,
                    pickled=pickled,
                )
                calls['warmup'] = lb.Call(_consume_warmup, self, warmup_iter)

        calls['open_radio'] = lb.Call(self.open_radio, sweep_spec.radio_setup)

        util.concurrently_with_fg(calls)

    def iter_sweep(
        self,
        sweep: specs.Sweep,
        calibration: 'xr.Dataset' = None,
        *,
        always_yield: bool = False,
        quiet: bool = False,
        pickled: bool = False,
        loop: bool = False,
        prepare: bool = False,
        reuse_compatible_iq: bool = False,
    ) -> sweeps.SweepIterator:
        # take args {3,4...N}
        kwargs = dict(locals())
        del kwargs['self'], kwargs['prepare']

        if prepare:
            self.warmup_sweep(sweep, calibration, pickled=False)

        radio = self.open_radio(sweep.radio_setup)
        radio.setup(sweep.radio_setup)

        return sweeps.iter_sweep(radio, **kwargs)

    def iter_raw_iq(
        self,
        sweep: specs.Sweep,
        calibration: 'xr.Dataset' = None,
        always_yield: bool = False,
        quiet: bool = False,
        pickled: bool = False,
        prepare: bool = True,
    ) -> typing.Generator['xr.Dataset']:
        # take args {3,4...N}
        kwargs = dict(locals())
        del kwargs['self'], kwargs['prepare']

        if prepare:
            prep_msg = self._describe_preparation(sweep)
            if prep_msg:
                lb.logger.info(prep_msg)
            self.warmup_sweep(sweep, calibration, pickled=True)

        radio = self.open_radio(sweep.radio_setup)
        radio.setup(sweep.radio_setup)

        return sweeps.iter_raw_iq(radio, sweep)


class _ServerService(rpyc.Service, SweepController):
    """API exposed by a server to remote clients"""

    def on_connect(self, conn: rpyc.Service):
        info = repr(conn._channel.stream.sock)
        try:
            source = eval(info[1:-1].split('raddr=', 1)[1])
        except IndexError:
            source = 'unknown address'
        lb.logger.info(f'new client connection from {source}')

    def on_disconnect(self, conn: rpyc.Service):
        info = repr(conn._channel.stream.sock)
        try:
            source = eval(info[1:-1].split('raddr=', 1)[1])
        except IndexError:
            source = 'unknown address'

        lb.logger.info(f'client at {source} disconnected')

    def exposed_radio_id(self, driver_name: str) -> str:
        return self.radios[driver_name].id

    def exposed_iter_sweep(
        self,
        sweep: specs.Sweep,
        calibration: 'xr.Dataset' = None,
        *,
        loop: bool = False,
        always_yield: bool = False,
        reuse_compatible_iq: bool = False,
    ) -> typing.Generator['xr.Dataset']:
        """wraps actions.sweep_iter to run on the remote server.

        For clients, rpyc exposes this in a connection object `conn` as as `conn.root.iter_sweep`.

        The calibrations dictionary maps filenames on the client to calibration data so that it can be
        accessed by the server.
        """

        conn = sweep.____conn__
        sweep = rpyc.utils.classic.obtain(sweep)

        with lb.stopwatch(
            f'obtaining calibration data {str(sweep.radio_setup.calibration)}',
            threshold=10e-3,
        ):
            calibration = rpyc.utils.classic.obtain(calibration)

        prep_msg = self._describe_preparation(sweep)
        if prep_msg:
            conn.root.deliver(None, prep_msg)
            lb.logger.info(prep_msg)
        self.warmup_sweep(sweep, calibration, pickled=True)

        capture_pairs = util.zip_offsets(sweep.captures, (0, -1), fill=None)

        descs = (
            captures.describe_capture(c1, c2, index=i, count=len(sweep.captures))
            for i, (c1, c2) in enumerate(capture_pairs)
        )

        sweep_iter = self.iter_sweep(
            sweep,
            calibration,
            always_yield=always_yield,
            pickled=True,
            loop=loop,
            prepare=False,
            reuse_compatible_iq=reuse_compatible_iq,
        )

        desc_pairs = zip_longest(sweep_iter, descs, fillvalue=None)

        if sweep_iter == []:
            return []
        else:
            return (conn.root.deliver(r, d) for r, d in desc_pairs)

    def exposed_acquire(
        self,
        capture: specs.RadioCapture,
        next_capture: typing.Union[specs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple['np.array', 'pd.Timestamp']:
        iq = self.radio.acquire(capture, next_capture, correction)

        if self.conn is None:
            raise RuntimeError('not connected')

        return self.conn.root.remote_shared_array(iq.shm_info())

    def exposed_read_stream(self, samples: int):
        return self.radio._read_stream(samples)

    def exposed_close_radio(self, radio_setup: specs.RadioSetup = None):
        radio_setup = rpyc.utils.classic.obtain(radio_setup)
        self.close_radio(radio_setup)


class _ClientService(rpyc.Service):
    """API exposed to a server by clients"""

    def on_connect(self, conn: rpyc.Service):
        lb.logger.info('connected to server')

    def on_disconnect(self, conn: rpyc.Service):
        lb.logger.info('disconnected from server')

    def exposed_deliver(
        self, pickled_dataset: 'xr.Dataset', description: str | None = None
    ):
        """serialize an object back to the client via pickling"""
        if description is not None:
            lb.logger.info(f'{description}')
        with lb.stopwatch('data transfer', threshold=10e-3, logger_level='debug'):
            if pickled_dataset is None:
                return None
            elif isinstance(pickled_dataset, bytes):
                return pickle.loads(pickled_dataset)
            elif isinstance(pickled_dataset, str):
                print(pickled_dataset, description)
                raise TypeError('expected pickle bytes but got str')

    def exposed_remote_shared_array(self, info):
        info = rpyc.utils.classic.obtain(info)
        return util.reference_shared_array(info)


def start_server(host=None, port=4567, default_driver: str | None = None):
    """start a server to run on a sensor (blocking)"""

    if default_driver is None:
        default_setup = None
    else:
        default_setup = specs.RadioSetup(driver=default_driver)

    t = rpyc.ThreadedServer(
        _ServerService(default_setup),
        hostname=host,
        port=port,
        protocol_config={'logger': lb.logger, 'allow_pickle': True},
    )
    lb.logger.info(f'hosting at {t.host}:{t.port}')
    t.start()


def connect(host='localhost', port=4567) -> rpyc.Connection:
    """connect to a remote sensor sensor.

    The returned connection object contains a `root` attribute that
    exposes remote wrappers for `striqt.sensor.SweepIterator`.

    Example::

        remote = connect_to_sensor('localhost')
        remote.root.iter_sweep()
    """

    # support host:port
    host, *extra = host.split(',', 1)
    if len(extra) != 0:
        port = int(extra[0])

    return rpyc.connect(
        host=host,
        port=port,
        config={'logger': lb.logger, 'allow_pickle': True},
        service=_ClientService,
    )
