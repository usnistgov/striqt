"""a SoapySource binding for the fake SoapySDR module in tests/sensor/fake_soapy.py.

Loaded through the `extensions:` block of the fake_soapy*.yaml sweeps. The source
spec mirrors the Airstack defaults except that it runs on numpy.
"""

from __future__ import annotations

from striqt.sensor import bindings, peripherals, sources, specs


class FakeSoapySourceSpec(specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: specs.types.MasterClockRate = 125e6
    array_backend: specs.types.ArrayBackend = 'numpy'
    receive_retries: specs.types.ReceiveRetries = 3

    rx_enable_delay = 0.35
    transient_holdoff_time = 2e-3
    stream_all_rx_ports = True


fake_soapy = bindings.bind_sensor(
    'fake_soapy',
    bindings.Sensor(
        source_cls=sources.SoapySource, peripherals_cls=peripherals.NoPeripherals
    ),
    bindings.Schema(
        source=FakeSoapySourceSpec,
        init_like=FakeSoapySourceSpec,
        capture=specs.SoapyCapture,
        arm_like=specs.SoapyCapture,
        peripherals=specs.Peripherals,
    ),
)

fake_soapy_calibration = bindings.bind_manual_yfactor_calibration(
    'fake_soapy_calibration', fake_soapy
)
