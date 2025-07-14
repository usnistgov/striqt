from __future__ import annotations
import typing

from labbench import paramattr as attr

from ..lib.sources import soapy
from ..lib import util


if typing.TYPE_CHECKING:
    import psutil
else:
    psutil = util.lazy_import('psutil')


# for TX only (RX channel is accessed through the AirT7201B.channel method)
channel_kwarg = attr.method_kwarg.int('channel', min=0, help='hardware port number')


class Air7x01B(soapy.SoapyRadioSource):
    resource = attr.value.dict({}, inherit=True)

    # adjust bounds based on the hardware
    lo_offset = attr.value.float(0.0, min=-125e6, max=125e6, inherit=True)
    lo_frequency = attr.method.float(min=300e6, max=6000e6, inherit=True)
    backend_sample_rate = attr.method.float(min=3.906250e6, max=125e6, inherit=True)
    gain = type(soapy.SoapyRadioSource.gain)(min=-30, max=0, step=0.5, inherit=True)
    tx_gain = attr.method.float(min=-41.95, max=0, step=0.1, inherit=True)
    rx_channel_count = attr.value.int(2, inherit=True)

    # set based on gain setting sweep tests
    _transient_holdoff_time = attr.value.float(2e-3, inherit=True)

    # stream setup and teardown for channel configuration are slow;
    # instead, stream all RX channels
    _stream_all_rx_channels = attr.value.bool(True, inherit=True)

    # without this, multichannel acquisition start time will vary
    # across channels, resulting in streaming errors
    _rx_enable_delay = attr.value.float(0.31, inherit=True)

    # float32 or int16: gpu work vs memory bandwidth tradeoff
    _transport_dtype = attr.value.str('int16', inherit=True)

    def open(self):
        # in some cases specifying the driver has caused exceptions on connect
        # validate it after the fact instead
        driver = self.backend.getDriverKey()
        if driver != 'SoapyAIRT':
            raise IOError(f'connected to {driver}, but expected SoapyAirT')

    def _post_connect(self):
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
        addr = 0x00040010
        start_bit = 8
        field_size = 4
        bit_range = range(start_bit, start_bit + field_size)
        field_mask = 0
        for bit in bit_range:
            field_mask |= 1 << bit

        # Read curr value
        reg = self.backend.readRegister('FPGA', addr)
        # Clear the bit field
        reg &= ~field_mask
        # Set values of mask, dropping extra bits
        field_val_mask = (value << start_bit) & field_mask

        # Set the bits
        reg |= field_val_mask
        # Write reg back
        self.backend.writeRegister('FPGA', addr, reg)

    @attr.property.str(inherit=True)
    def id(self):
        # as of 1.0.0, AirStack doesn't seem to return a serial through Soapy
        # instead, take the Jetson ethernet MAC address as the radio id.
        try:
            if_addrs = psutil.net_if_addrs()['eth0']
        except KeyError:
            raise OSError('no eth0 to create radio_id')

        for snic_addr in if_addrs:
            if snic_addr.address[2] == ':':
                eth0_mac = snic_addr.address.replace(':', '')
                break
        else:
            raise OSError('no MAC address reported for the eth0 interface')

        return eth0_mac

    def get_temperatures(self) -> dict[str, float]:
        """returns the transceiver temperature in Celsius"""

        return {'radio_transceiver': self.backend.readSensorFloat('xcvr_temp')}


class AirT7x01B(Air7x01B):
    # for backward compatibility
    pass


class Air7201B(Air7x01B):
    pass


class Air7101B(Air7x01B):
    pass


if __name__ == '__main__':
    airt = Air7201B(freq=2.44e9, fs=2 * 31.25e6)
    iq, *_ = airt.acquire(256 * 1024)
    airt.close()
