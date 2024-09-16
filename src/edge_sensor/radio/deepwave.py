from __future__ import annotations
import labbench.paramattr as attr

from .soapy import SoapyRadioDevice
import uuid

# for TX only (RX channel is accessed through the AirT7201B.channel method)
channel_kwarg = attr.method_kwarg.int(
    'channel', min=0, max=1, help='hardware port number'
)


class _AirT7x01B(SoapyRadioDevice):
    resource = attr.value.dict(default={}, inherit=True)

    # adjust bounds based on the hardware
    duration = attr.value.float(100e-3, inherit=True)
    lo_offset = attr.value.float(0.0, min=-125e6, max=125e6, inherit=True)
    channel = attr.method.int(min=0, max=1, inherit=True)
    lo_frequency = attr.method.float(min=300e6, max=6000e6, inherit=True)
    backend_sample_rate = attr.method.float(min=3.906250e6, max=125e6, inherit=True)
    gain = attr.method.float(min=-30, max=0, step=0.5, inherit=True)
    tx_gain = attr.method.float(min=-41.95, max=0, step=0.1, inherit=True)

    def open(self):
        # in some cases specifying the driver has caused exceptions on connect
        # validate it after the fact instead
        driver = self.backend.getDriverKey()
        if driver != 'SoapyAIRT':
            raise IOError(f'connected to {driver}, but expected SoapyAirT')

    @attr.property.str(inherit=True)
    def id(self):
        # Jetson UUID - AirStack doesn't seem to return serial through Soapy
        return hex(uuid.getnode())


class Air7201B(_AirT7x01B):
    pass


class Air7101B(_AirT7x01B):
    pass


if __name__ == '__main__':
    airt = Air7201B(freq=2.44e9, fs=2 * 31.25e6)
    iq = airt.acquire(256 * 1024)
    airt.close()
