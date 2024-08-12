from __future__ import annotations

import rpyc
import labbench as lb
import xarray as xr
import sys

from typing import Generator, Optional, Any

from edge_sensor import actions
from edge_sensor.structs import Sweep, RadioCapture, RadioSetup
from edge_sensor.radio import find_radio_cls_by_name, RadioDevice
from edge_sensor.radio.util import is_same_resource
from edge_sensor.util import set_cuda_mem_limit, zip_offsets

_PROTOCOL_CONFIG = {'logger': lb.logger, 'allow_pickle': True}


class SweepController:
    """Manage local edge sensor operation, encapsulating radio connection logic.
    
    This is also used by `start_sensor_server` to serve remote operations.
    """

    def __init__(self, radio_setup: RadioSetup = None):
        self.radios: dict[str, RadioDevice] = {}
        self.prepared_sweeps: dict[str, RadioCapture] = {}
        self.handlers: dict[rpyc.Connection, Any] = {}
        set_cuda_mem_limit()

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
        resource = radio_setup.resource

        if driver_name in self.radios and self.radios[driver_name].isopen:
            if is_same_resource(self.radios[driver_name].resource, radio_setup.resource):
                lb.logger.debug(f'reusing open {repr(driver_name)}')
                return self.radios[driver_name]
            else:
                lb.logger.debug(f're-opening {repr(driver_name)} to set resource={repr(resource)}')
                self.radios[driver_name].close()
        else:
            lb.logger.debug(f'opening driver {repr(driver_name)}')

        radio_cls = find_radio_cls_by_name(driver_name)
        radio = self.radios[driver_name] = radio_cls()
        if resource is not None:
            radio.resource = resource
        radio.open()

        return radio

    def iter_sweep(self, sweep_spec: Sweep, swept_fields: list[str]) -> Generator[xr.Dataset]:
        radio = self.open_radio(sweep_spec.radio_setup)
        radio.setup(sweep_spec.radio_setup)
        return actions.iter_sweep(radio, sweep_spec, swept_fields)

    def __del__(self):
        self.close()


class _ControllerService(rpyc.Service, SweepController):
    """this is what is exposed by a server to remote clients"""

    def exposed_iter_sweep(self, sweep_spec: Sweep, swept_fields: list[str]) -> Generator[xr.Dataset]:
        """wraps actions.sweep_iter to run on the remote server.

        rpyc mangles attribute names to access this when remote clients call `conn.root.iter_sweep`.
        """

        conn = sweep_spec.____conn__
        sweep_spec = rpyc.utils.classic.obtain(sweep_spec)
        swept_fields = rpyc.utils.classic.obtain(swept_fields)

        descs = [actions.describe_capture(c, swept_fields) for c in sweep_spec.captures]
        
        return (
            conn.root.deliver(r, d)
            for r,d in zip(self.iter_sweep(sweep_spec, swept_fields), descs)
        )


class _ClientService(rpyc.Service):
    def exposed_deliver(self, dataset: xr.Dataset, description:Optional[str]=None):
        """serialize an object back to the client via pickling"""
        if description is not None:
            lb.logger.info(f'capture • {description}')
        with lb.stopwatch('data transfer', logger_level='debug'):
            return rpyc.utils.classic.obtain(dataset)


def start_server(host=None, port=4567, default_driver:Optional[str] = None):
    """start a server to run on a sensor (blocking)"""

    if default_driver is None:
        default_setup = None
    else:
        default_setup = RadioSetup(driver=default_driver)

    t = rpyc.ThreadedServer(
        _ControllerService(default_setup),
        hostname=host,
        port=port,
        protocol_config=_PROTOCOL_CONFIG
    )
    lb.logger.info(f'starting server hosting at {t.host}:{t.port}')
    t.start()


def connect(host='localhost', port=4567) -> rpyc.Connection:
    """connect to a remote sensor sensor.
    
    The returned connection object contains a `root` attribute that
    exposes remote wrappers for `edge_sensor.actions`.

    Example::

        remote = connect_to_sensor('localhost')
        remote.root.iter_sweep()
    """

    return rpyc.connect(host=host, port=port, config=_PROTOCOL_CONFIG, service=_ClientService)