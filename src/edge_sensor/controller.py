from __future__ import annotations

import rpyc
import labbench as lb

import typing
from typing import Generator, Optional, Any

from channel_analysis import type_stubs
from edge_sensor import actions, util
from edge_sensor.structs import Sweep, RadioCapture, RadioSetup, describe_capture
from edge_sensor.radio import find_radio_cls_by_name, is_same_resource, RadioDevice

if typing.TYPE_CHECKING:
    import xarray as xr
else:
    xr = lb.util.lazy_import('xarray')


_PROTOCOL_CONFIG = {'logger': lb.logger, 'allow_pickle': True}


class SweepController:
    """Manage local edge sensor operation, encapsulating radio connection logic.

    This is also used by `start_sensor_server` to serve remote operations.
    """

    def __init__(self, radio_setup: RadioSetup = None):
        self.radios: dict[str, RadioDevice] = {}
        self.warmed_captures: set[RadioCapture] = set()
        self.handlers: dict[rpyc.Connection, Any] = {}
        util.set_cuda_mem_limit()

        if radio_setup is not None:
            self.open_radio(radio_setup)

    def close(self):
        last_ex = None

        for radio in self.radios.values():
            try:
                radio.close()
            except BaseException as ex:
                last_ex = ex

        if last_ex is not None:
            raise last_ex

    def open_radio(self, radio_setup: RadioSetup):
        driver_name = radio_setup.driver
        radio_cls = find_radio_cls_by_name(driver_name)

        if radio_setup.resource is None:
            resource = radio_cls.resource.default
        else:
            resource = radio_setup.resource

        if driver_name in self.radios and self.radios[driver_name].isopen:
            if is_same_resource(self.radios[driver_name].resource, resource):
                lb.logger.debug(f'reusing open {repr(driver_name)}')
                return self.radios[driver_name]
            else:
                lb.logger.debug(
                    f're-opening {repr(driver_name)} to set resource={repr(resource)}'
                )
                self.radios[driver_name].close()
        else:
            lb.logger.debug(f'opening driver {repr(driver_name)}')

        radio = self.radios[driver_name] = radio_cls()
        if resource is not None:
            radio.resource = resource
        radio.open()

        return radio

    def close_radio(self, radio_setup: RadioSetup):
        self.radios[radio_setup.driver].close()

    def _describe_preparation(self, sweep: Sweep) -> str:
        warmup_sweep = actions.design_warmup_sweep(
            sweep, skip=tuple(self.warmed_captures)
        )
        msgs = []
        if sweep.radio_setup.driver not in self.radios:
            msgs += [f'opening {sweep.radio_setup.driver} radio']
        if len(warmup_sweep.captures) > 0:
            msgs += [
                f'warming GPU DSP with {len(warmup_sweep.captures)} empty captures'
            ]
        return ' and '.join(msgs)

    def prepare_sweep(self, sweep_spec: Sweep, swept_fields: list[str], calibration):
        """open the radio while warming up the GPU"""

        warmup_sweep = actions.design_warmup_sweep(
            sweep_spec, skip=tuple(self.warmed_captures)
        )
        self.warmed_captures = self.warmed_captures | set(warmup_sweep.captures)

        if len(warmup_sweep.captures) > 0:
            warmup_iter = self.iter_sweep(warmup_sweep, swept_fields, calibration)
        else:
            return []

        lb.concurrently(
            warmup=lb.Call(list, warmup_iter),
            open_radio=lb.Call(self.open_radio, sweep_spec.radio_setup),
        )

        self.close_radio(warmup_sweep)

    def iter_sweep(
        self,
        sweep_spec: Sweep,
        swept_fields: list[str],
        calibration: type_stubs.DatasetType = None,
        always_yield: bool = False,
    ) -> Generator[xr.Dataset]:
        radio = self.open_radio(sweep_spec.radio_setup)
        radio.setup(sweep_spec.radio_setup)

        return actions.iter_sweep(
            radio, sweep_spec, swept_fields, calibration, always_yield
        )

    def __del__(self):
        self.close()


class _ServerService(rpyc.Service, SweepController):
    """API exposed by a server to remote clients"""

    def on_connect(self, conn: rpyc.Service):
        lb.logger.info('connected to client')

    def on_disconnect(self, conn: rpyc.Service):
        lb.logger.info('disconnected from client')

    def exposed_iter_sweep(
        self,
        sweep_spec: Sweep,
        swept_fields: list[str],
        calibration: type_stubs.DatasetType = None,
        always_yield: bool = False,
    ) -> Generator[xr.Dataset]:
        """wraps actions.sweep_iter to run on the remote server.

        For clients, rpyc exposes this in a connection object `conn` as as `conn.root.iter_sweep`.

        The calibrations dictionary maps filenames on the client to calibration data so that it can be
        accessed by the server.
        """

        conn = sweep_spec.____conn__
        sweep_spec = rpyc.utils.classic.obtain(sweep_spec)
        swept_fields = rpyc.utils.classic.obtain(swept_fields)

        with lb.stopwatch(
            f'obtaining calibration data {str(sweep_spec.radio_setup.calibration)}'
        ):
            calibration = rpyc.utils.classic.obtain(calibration)

        prep_msg = self._describe_preparation(sweep_spec)
        if prep_msg:
            conn.root.deliver(None, prep_msg)
            lb.logger.info(prep_msg)
        self.prepare_sweep(sweep_spec, swept_fields, calibration)

        descs = (
            f'{i+1}/{len(sweep_spec.captures)} {describe_capture(c, swept_fields)}'
            for i, c in enumerate(sweep_spec.captures)
        )

        generator = self.iter_sweep(sweep_spec, swept_fields, calibration, always_yield)

        if generator == []:
            return []
        else:
            return (conn.root.deliver(r, d) for r, d in zip(generator, descs))


class _ClientService(rpyc.Service):
    """API exposed to a server by clients"""

    def on_connect(self, conn: rpyc.Service):
        lb.logger.info('connected to server')

    def on_disconnect(self, conn: rpyc.Service):
        lb.logger.info('disconnected from server')

    def exposed_deliver(
        self, dataset: type_stubs.DatasetType, description: Optional[str] = None
    ):
        """serialize an object back to the client via pickling"""
        if description is not None:
            lb.logger.info(f'{description}')
        with lb.stopwatch('data transfer', logger_level='debug'):
            return rpyc.utils.classic.obtain(dataset)


def start_server(host=None, port=4567, default_driver: Optional[str] = None):
    """start a server to run on a sensor (blocking)"""

    if default_driver is None:
        default_setup = None
    else:
        default_setup = RadioSetup(driver=default_driver)

    t = rpyc.ThreadedServer(
        _ServerService(default_setup),
        hostname=host,
        port=port,
        protocol_config=_PROTOCOL_CONFIG,
    )
    lb.logger.info(f'hosting at {t.host}:{t.port}')
    t.start()


def connect(host='localhost', port=4567) -> rpyc.Connection:
    """connect to a remote sensor sensor.

    The returned connection object contains a `root` attribute that
    exposes remote wrappers for `edge_sensor.actions`.

    Example::

        remote = connect_to_sensor('localhost')
        remote.root.iter_sweep()
    """

    return rpyc.connect(
        host=host, port=port, config=_PROTOCOL_CONFIG, service=_ClientService
    )
