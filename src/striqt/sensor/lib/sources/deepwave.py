from __future__ import annotations

import functools
import typing

from .. import specs, util
from . import soapy

if typing.TYPE_CHECKING:
    import psutil
else:
    psutil = util.lazy_import('psutil')


class Air7101BSourceSpec(
    specs.SoapySource,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    base_clock_rate: specs.BaseClockRateType = 125e6
    rx_enable_delay = 0.35
    transient_holdoff_time = 2e-3


class Air7201BSourceSpec(
    specs.SoapySource,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    base_clock_rate: specs.BaseClockRateType = 125e6
    rx_enable_delay = 0.35
    transient_holdoff_time = 2e-3


class Air8201BSourceSpec(
    specs.SoapySource,
    forbid_unknown_fields=True,
    frozen=True,
    cache_hash=True,
    kw_only=True,
):
    base_clock_rate: specs.BaseClockRateType = 125e6
    rx_enable_delay = 0.4
    transient_holdoff_time = 2e-3


class Airstack1Source(soapy.SoapySourceBase):
    def _connect(self, spec: specs.SoapySource):
        super()._connect(spec)
        assert self._device is not None

        driver = self._device.getDriverKey()
        if driver != 'SoapyAIRT':
            raise IOError(f'connected to {driver}, but expected SoapyAirT')

        self._set_jesd_sysref_delay(0)

    def _set_jesd_sysref_delay(self, value: int):
        """
        SYSREF delay: add additional delay to SYSREF re-alignment of LMFC counter
        1111 = 15 core_clk cycles delay
        ....
        0000 = 0 core_clk cycles delay
        In order to move away from the LFMC rollover we need to set bits 11:8
        of the SYSREF handling register which is at address 0x0004_0010.
        This register needs to be set before we try to sync the JESD204B bus.

        Ref: https://docs.deepwavedigital.com/Tutorials/8_triggered_signal_stream/#maintaining-fixed-delay-between-calibrations
        """
        assert self._device is not None
        addr = 0x00040010
        start_bit = 8
        field_size = 4
        bit_range = range(start_bit, start_bit + field_size)
        field_mask = 0
        for bit in bit_range:
            field_mask |= 1 << bit

        # Read curr value
        reg = self._device.readRegister('FPGA', addr)
        # Clear the bit field
        reg &= ~field_mask
        # Set values of mask, dropping extra bits
        field_val_mask = (value << start_bit) & field_mask

        # Set the bits
        reg |= field_val_mask
        # Write reg back
        self._device.writeRegister('FPGA', addr, reg)

    @functools.cached_property
    def id(self) -> str:
        # as of 1.0.0, AirStack doesn't seem to return a serial through Soapy
        # instead, take the Jetson ethernet MAC address as the radio id.
        try:
            if_addrs = psutil.net_if_addrs()['eth0']
        except KeyError:
            raise OSError('no eth0 to create source_id')

        for snic_addr in if_addrs:
            if snic_addr.address[2] == ':':
                eth0_mac = snic_addr.address.replace(':', '')
                break
        else:
            raise OSError('no MAC address reported for the eth0 interface')

        return eth0_mac

    def read_peripherals(self) -> dict[str, float]:
        """returns the transceiver temperature in Celsius"""
        assert self._device is not None

        return {'transceiver': self._device.readSensorFloat('xcvr_temp')}
