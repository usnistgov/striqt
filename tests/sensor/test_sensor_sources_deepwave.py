"""striqt.sensor.lib.sources.deepwave: Airstack source specs, their bindings, the JESD
SYSREF register field, and the MAC-address radio id, against the fake SoapySDR device
and (under the `hardware` marker) an attached radio"""

from __future__ import annotations

import types

import numpy as np
import pytest

import striqt.sensor as ss
from striqt.sensor.lib.sources import deepwave, soapy

SPECS = {
    'air7101b': deepwave.Air7101BSourceSpec,
    'air7201b': deepwave.Air7201BSourceSpec,
    'air8201b': deepwave.Air8201BSourceSpec,
}

# %% spec defaults


@pytest.mark.parametrize('cls', list(SPECS.values()), ids=list(SPECS))
def test_airstack_specs_target_the_gpu_with_retries(cls):
    spec = cls()
    assert spec.master_clock_rate == pytest.approx(125e6)
    assert spec.array_backend == 'cupy'
    assert spec.receive_retries == 3
    assert spec.stream_all_rx_ports is True
    assert spec.transport_dtype == 'float32'
    assert isinstance(spec, ss.specs.SoapySource)


@pytest.mark.parametrize('name', list(SPECS))
def test_bindings_pair_source_specs(name):
    ctrl = getattr(ss.bindings, name)
    assert ctrl.schema.source is SPECS[name]
    assert ctrl.sensor.source_cls is deepwave.Airstack1Source
    assert ctrl.schema.capture is ss.specs.SoapyCapture


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='bindings.py binds air7201b with init_like=Air7101BSourceSpec',
)
def test_air7201b_init_like_matches_its_source_spec():
    assert ss.bindings.air7201b.schema.init_like is deepwave.Air7201BSourceSpec


# %% _set_jesd_sysref_delay


class _Registers:
    def __init__(self, preset: int):
        self.value = preset
        self.calls = []

    def readRegister(self, name, addr):
        self.calls.append(('read', name, addr))
        return self.value

    def writeRegister(self, name, addr, value):
        self.calls.append(('write', name, addr))
        self.value = value


@pytest.mark.parametrize(
    'preset, value, expected',
    [
        (0xFFFF, 0, 0xF0FF),
        (0xFFFF, 5, 0xF5FF),
        (0x0000, 15, 0x0F00),
        # only bits 11:8 belong to the field
        (0xFFFF, 0x10, 0xF0FF),
    ],
)
def test_sysref_delay_sets_only_bits_11_to_8(preset, value, expected):
    registers = _Registers(preset)
    source = types.SimpleNamespace(device=registers)

    deepwave.Airstack1Source._set_jesd_sysref_delay(source, value)

    assert registers.value == expected
    assert registers.calls == [
        ('read', 'FPGA', 0x00040010),
        ('write', 'FPGA', 0x00040010),
    ]


# %% get_id


def _snic(address):
    return types.SimpleNamespace(address=address)


@pytest.fixture
def net_if_addrs(monkeypatch):
    table = {}
    monkeypatch.setattr(deepwave.psutil, 'net_if_addrs', lambda: table)
    return table


def test_get_id_is_the_eth0_mac_without_separators(net_if_addrs):
    net_if_addrs['eth0'] = [_snic('192.168.0.2'), _snic('48:b0:2d:3e:4f:50')]
    assert deepwave.Airstack1Source.get_id(None) == '48b02d3e4f50'


def test_get_id_without_eth0(net_if_addrs):
    net_if_addrs['wlan0'] = [_snic('48:b0:2d:3e:4f:50')]
    with pytest.raises(OSError, match='no eth0'):
        deepwave.Airstack1Source.get_id(None)


def test_get_id_without_a_mac_address(net_if_addrs):
    net_if_addrs['eth0'] = [_snic('192.168.0.2'), _snic('fe80::1')]
    with pytest.raises(OSError, match='no MAC address'):
        deepwave.Airstack1Source.get_id(None)


# %% Airstack1Source (fake SoapySDR device)


class TestAirstack1Source:
    def test_opens_with_the_airt_driver_and_clock_settings(self, fake_soapy):
        spec = deepwave.Air7101BSourceSpec(
            array_backend='numpy', time_source='gps', clock_source='external'
        )
        source = deepwave.Airstack1Source(spec)

        assert source.device.kwargs == {
            'driver': 'SoapyAIRT',
            'time_src': 'gps',
            'clk_src': 'external',
        }
        assert source.get_info().driver == 'SoapyAIRT'

    def test_host_time_source_maps_to_internal(self, fake_soapy):
        spec = deepwave.Air7101BSourceSpec(array_backend='numpy', time_source='host')
        source = deepwave.Airstack1Source(spec)
        assert source.device.kwargs['time_src'] == 'internal'

    def test_open_clears_the_sysref_delay_field(self, fake_soapy):
        spec = deepwave.Air7101BSourceSpec(array_backend='numpy')
        source = deepwave.Airstack1Source(spec)
        device = source.device
        assert device.calls_named('writeRegister') == [
            ('writeRegister', 'FPGA', 0x00040010, 0)
        ]

        device.registers['FPGA', 0x00040010] = 0xFFFF
        source._set_jesd_sysref_delay(3)
        assert device.registers['FPGA', 0x00040010] == 0xF3FF

    def test_read_peripherals_reports_the_transceiver_temperature(self, fake_soapy):
        source = deepwave.Airstack1Source(
            deepwave.Air7101BSourceSpec(array_backend='numpy')
        )
        assert source.read_peripherals() == {'transceiver': 41.5}


# %% real hardware: an Airstack radio (Jetson, STRIQT_TEST_HARDWARE=1)

AIRSTACK_CAPTURE = {
    'port': (0, 1),
    'center_frequency': 3.75e9,
    'gain': (0, 0),
    'duration': 1e-3,
    'sample_rate': 125e6,
    'host_resample': False,
}


@pytest.fixture(scope='module')
def airstack_source_spec():
    """the air7101b source spec, skipped unless striqt imported the real SoapySDR
    and a radio answers the enumeration"""
    pytest.importorskip('SoapySDR')
    if (
        soapy.SoapySDR is None
        or getattr(soapy.SoapySDR, '__name__', None) != 'SoapySDR'
    ):
        pytest.skip('striqt did not import SoapySDR at load time')
    if not soapy.SoapySDR.Device.enumerate({'driver': 'SoapyAIRT'}):
        pytest.skip('no SoapyAIRT device is attached')
    return ss.bindings.air7101b.schema.source(array_backend='numpy')


@pytest.mark.hardware
class TestAirstackHardware:
    def test_probe(self, airstack_source_spec):
        with ss.bindings.air7101b.from_source_spec(airstack_source_spec) as ctrl:
            info = ctrl.source_info
            assert info.driver == 'SoapyAIRT'
            assert info.num_rx_ports == 2
            assert info.has_timestamps
            assert 'FPGA' in info.registers

    def test_open_clears_the_sysref_delay_field(self, airstack_source_spec):
        with ss.bindings.air7101b.from_source_spec(airstack_source_spec) as ctrl:
            register = ctrl.backend.device.readRegister('FPGA', 0x00040010)
            assert register & 0x0F00 == 0

    def test_id_is_the_eth0_mac(self, airstack_source_spec):
        with ss.bindings.air7101b.from_source_spec(airstack_source_spec) as ctrl:
            assert len(ctrl.source_id) == 12
            int(ctrl.source_id, 16)

    def test_transceiver_temperature_is_plausible(self, airstack_source_spec):
        with ss.bindings.air7101b.from_source_spec(airstack_source_spec) as ctrl:
            temperature = ctrl.backend.read_peripherals()['transceiver']
            assert 10 < temperature < 90

    def test_host_time_sync_lands_near_the_host_clock(self, airstack_source_spec):
        spec = airstack_source_spec.replace(time_source='host')
        with ss.bindings.air7101b.from_source_spec(spec) as ctrl:
            device = ctrl.backend.device
            ctrl.backend.sync_time(device)
            hardware = device.getHardwareTime('now') / 1e9
            assert hardware == pytest.approx(soapy.time.time(), abs=0.2)

    def test_acquire_two_ports(self, airstack_source_spec):
        capture = ss.specs.SoapyCapture(**AIRSTACK_CAPTURE)
        with ss.bindings.air7101b.from_source_spec(
            airstack_source_spec, rx_ports=(0, 1)
        ) as ctrl:
            ctrl._arm_spec(capture)
            first = ctrl.acquire()
            second = ctrl.acquire()

        overlaps = ss.lib.compute.get_correction_overlaps(capture, airstack_source_spec)
        assert first.pre_align.shape == (2, 125_000 + sum(overlaps))
        assert first.pre_align.dtype == np.complex64
        assert np.isfinite(first.pre_align).all()
        assert np.abs(first.pre_align).max() <= 1.0
        assert first.info.backend_sample_rate == pytest.approx(125e6)

        host_now = soapy.time.time()
        assert abs(first.info.start_time.timestamp() - host_now) < 5
        assert second.info.start_time - first.info.start_time >= np.timedelta64(1, 'ms')
