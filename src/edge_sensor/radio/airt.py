from __future__ import annotations
import labbench.paramattr as attr

from .soapy import SoapyRadioDevice

# for TX only (RX channel is accessed through the AirT7201B.channel method)
channel_kwarg = attr.method_kwarg.int(
    'channel', min=0, max=1, help='hardware port number'
)

class AirT7201B(SoapyRadioDevice):
    resource = attr.value.dict(
        default={'driver': 'SoapyAIRT'},
        inherit=True
    )

    # adjust bounds based on the hardware
    duration = attr.value.float(100e-3, inherit=True)
    lo_offset = attr.value.float(0.0, min=-125e6, max=125e6, inherit=True)
    channel = attr.method.int(min=0, max=1, inherit=True)
    lo_frequency = attr.method.float(min=300e6, max=6000e6, inherit=True)
    backend_sample_rate = attr.method.float(min=3.906250e6, max=125e6, inherit=True)
    gain = attr.method.float(min=-30, max=0, step=0.5, inherit=True)
    tx_gain = attr.method.float(min=-41.95, max=0, step=0.1, inherit=True)

    # TODO: do the expressions handle inheritance properly?
    # sample_rate = backend_sample_rate.corrected_from_expression(
    #     backend_sample_rate / SoapyRadioDevice._downsample,
    #     label='Hz',
    #     help='sample rate of acquired waveform',
    # )


if __name__ == '__main__':
    airt = AirT7201B(freq=2.44e9, fs=2 * 31.25e6)
    iq = airt.acquire(256 * 1024)
    airt.close()
