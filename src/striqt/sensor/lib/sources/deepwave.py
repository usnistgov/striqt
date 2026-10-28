from __future__ import annotations as __

import typing

from ... import specs
from .. import util
from . import soapy

if typing.TYPE_CHECKING:
    import psutil
else:
    psutil = util.lazy_import('psutil')


class Air7101BSourceSpec(specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: specs.types.MasterClockRate = 125e6
    array_backend: specs.types.ArrayBackend = 'cupy'
    receive_retries: specs.types.ReceiveRetries = 3

    rx_enable_delay = 0.35
    transient_holdoff_time = 2e-3
    stream_all_rx_ports = True


class Air7201BSourceSpec(specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: specs.types.MasterClockRate = 125e6
    array_backend: specs.types.ArrayBackend = 'cupy'
    receive_retries: specs.types.ReceiveRetries = 3

    rx_enable_delay = 0.35
    transient_holdoff_time = 2e-3
    stream_all_rx_ports = True


class Air8201BSourceSpec(specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: specs.types.MasterClockRate = 125e6
    array_backend: specs.types.ArrayBackend = 'cupy'
    receive_retries: specs.types.ReceiveRetries = 3

    rx_enable_delay = 1.4
    transient_holdoff_time = 2e-3
    stream_all_rx_ports = True


class Airstack1Source(soapy.SoapySource):
    def __init__(self, spec: specs.SoapySource, **kwargs):
        # trim script startup time by setting these here
        air_kwargs = dict(
            kwargs,
            driver='SoapyAIRT',
            time_src=soapy.device_time_source(spec),
            clk_src=spec.clock_source,
        )

        super().__init__(spec, **air_kwargs)

        self._set_jesd_sysref_delay(0)

    def _set_jesd_sysref_delay(self, value: int):
        """set the SYSREF delay of the FPGA JESD204B SYSREF handling register.

        `value` is the delay (in core_clk cycles, 0 to 15) of the SYSREF
        re-alignment of the LMFC counter, written to bits 11:8 of the register at
        address ``0x0004_0010``. A nonzero delay moves the re-alignment away from the
        LMFC rollover. The register must be written before the JESD204B bus is
        synchronized.

        References:
            https://docs.deepwavedigital.com/Tutorials/8_triggered_signal_stream/#maintaining-fixed-delay-between-calibrations
        """
        assert self.device is not None
        addr = 0x00040010
        start_bit = 8
        field_size = 4
        bit_range = range(start_bit, start_bit + field_size)
        field_mask = 0
        for bit in bit_range:
            field_mask |= 1 << bit

        reg = self.device.readRegister('FPGA', addr)
        reg &= ~field_mask
        field_val_mask = (value << start_bit) & field_mask

        reg |= field_val_mask
        self.device.writeRegister('FPGA', addr, reg)

    def get_id(self) -> str:
        # AirStack 1.0.0 returns no serial through Soapy, so the Jetson eth0 MAC
        # address is the radio id
        try:
            if_addrs = psutil.net_if_addrs()['eth0']
        except KeyError:
            raise OSError('no eth0 network interface')

        for snic_addr in if_addrs:
            if snic_addr.address[2] == ':':
                eth0_mac = snic_addr.address.replace(':', '')
                break
        else:
            raise OSError('no MAC address reported for the eth0 interface')

        return eth0_mac

    def read_peripherals(self) -> dict[str, float]:
        """returns the transceiver temperature in Celsius"""
        assert self.device is not None

        return {'transceiver': self.device.readSensorFloat('xcvr_temp')}
