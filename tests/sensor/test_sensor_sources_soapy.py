"""striqt.sensor.lib.sources.soapy: the pieces that run without a SoapySDR module.
Stream reads, overload accounting, capability structs, time source mapping, and
calibration assignment onto acquired IQ"""

from __future__ import annotations

import types
from typing import NamedTuple

import msgspec
import numpy as np
import pytest
from sweep_strategies import BOLTZMANN_MW, receiver_gain, save_yfactor_calibration

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.compute import design_resampler
from striqt.sensor.lib.sources import soapy
from striqt.sensor.lib.sources.base import ReceiveStreamError

MCR = 125e6

# the subset of the SoapySDR module surface that the tested code touches; the values
# are the library's (SoapySDR/Errors.h)
SOAPY_CONSTANTS = {
    'SOAPY_SDR_TIMEOUT': -1,
    'SOAPY_SDR_STREAM_ERROR': -2,
    'SOAPY_SDR_CORRUPTION': -3,
    'SOAPY_SDR_OVERFLOW': -4,
    'SOAPY_SDR_NOT_SUPPORTED': -5,
    'SOAPY_SDR_TIME_ERROR': -6,
    'SOAPY_SDR_UNDERFLOW': -7,
}
ERROR_NAMES = {v: k.replace('SOAPY_SDR_', '') for k, v in SOAPY_CONSTANTS.items()}


class StreamResult(NamedTuple):
    ret: int
    flags: int = 0
    timeNs: int = 0
    chanMask: int = 0


class Range:
    """duck-types SoapySDR.Range"""

    def __init__(self, minimum, maximum, step=0.0):
        self._values = (minimum, maximum, step)

    def minimum(self):
        return self._values[0]

    def maximum(self):
        return self._values[1]

    def step(self):
        return self._values[2]


def arg_info(key='gain', **kws):
    """duck-types SoapySDR.ArgInfo"""
    fields = {
        'key': key,
        'name': key.title(),
        'description': f'the {key}',
        'units': 'dB',
        'type': 2,
        'value': '0',
        'range': Range(-30.0, 0.0, 0.5),
        'options': ('a', 'b'),
    }
    return types.SimpleNamespace(**dict(fields, **kws))


@pytest.fixture
def soapy_constants(monkeypatch):
    module = types.SimpleNamespace(
        **SOAPY_CONSTANTS, errToStr=lambda code: ERROR_NAMES[code]
    )
    monkeypatch.setattr(soapy, 'SoapySDR', module)
    return module


def _capture(**kws):
    kws = {
        'port': 0,
        'center_frequency': 1e9,
        'gain': 0,
        'sample_rate': MCR,
        'duration': 1e-3,
        'analysis_bandwidth': 40e6,
        'host_resample': False,
        **kws,
    }
    return ss.specs.SoapyCapture(**kws)


# %% _SoapyRange and _SoapyArgInfo


def test_range_from_soapy():
    r = soapy._SoapyRange.from_soapy(Range(-30.0, 0.0, 0.5))
    assert (r.minimum, r.maximum, r.step) == (-30.0, 0.0, 0.5)


def test_range_from_soapy_tuple():
    ranges = soapy._SoapyRange.from_soapy_tuple((Range(1.0, 2.0), Range(3.0, 4.0)))
    assert [r.maximum for r in ranges] == [2.0, 4.0]
    assert ranges[0].step == pytest.approx(0.0)


def test_arg_info_from_soapy():
    info = soapy._SoapyArgInfo.from_soapy(arg_info())
    assert info.name == 'Gain'
    assert info.range == (soapy._SoapyRange(minimum=-30.0, maximum=0.0, step=0.5),)
    assert info.options == ('a', 'b')


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_SoapyArgInfo.range is annotated as a bare tuple, so validate() leaves '
    'the _SoapyRange entries as dicts',
)
def test_arg_info_validate_round_trips():
    info = soapy._SoapyArgInfo.from_soapy(arg_info())
    assert info.validate() == info


def test_arg_info_map_is_keyed_by_the_soapy_key():
    infos = soapy._SoapyArgInfo.from_soapy_map([arg_info('gain'), arg_info('lna')])
    assert list(infos) == ['gain', 'lna']
    assert infos['lna'].description == 'the lna'


@pytest.mark.xfail(
    strict=True,
    raises=msgspec.ValidationError,
    reason='_SoapyPortInfo annotates full_freq_range and backend_sample_rate_range as '
    '_SoapyRange, but _probe_channel fills them from from_soapy_tuple',
)
def test_port_info_round_trips_as_probed():
    span = soapy._SoapyRange.from_soapy(Range(-30.0, 0.0, 0.5))
    info = soapy._SoapyPortInfo(
        port_info={},
        full_duplex=False,
        agc=False,
        stream_formats=('CF32',),
        stream_args={},
        antennas=('RX',),
        corrections=(),
        gains={'PGA': span},
        full_gain_range=span,
        frequencies={'RF': (span,)},
        full_freq_range=soapy._SoapyRange.from_soapy_tuple((Range(300e6, 6e9),)),
        tune_args={},
        backend_sample_rate_range=soapy._SoapyRange.from_soapy_tuple((
            Range(3.9e6, 125e6),
        )),
        master_clock_rates=(125e6,),
        bandwidths=(span,),
        sensors={},
        settings={},
    )
    assert info.validate() == info


# %% device_time_source


@pytest.mark.parametrize(
    'time_source, expected',
    [
        ('host', 'internal'),
        ('internal', 'internal'),
        ('external', 'external'),
        ('gps', 'gps'),
    ],
)
def test_device_time_source(time_source, expected):
    spec = ss.specs.SoapySource(master_clock_rate=MCR, time_source=time_source)
    assert soapy.device_time_source(spec) == expected


# %% RxStream.validate_stream_read

validate = soapy.RxStream.validate_stream_read


def test_successful_read_returns_count_and_timestamp():
    assert validate(StreamResult(ret=1024, timeNs=5_000), 'except') == (1024, 5_000)


def test_zero_length_read_is_reported_as_success():
    assert validate(StreamResult(ret=0, timeNs=7), 'except') == (0, 7)


def test_timestamp_after_sync_is_accepted():
    assert validate(StreamResult(ret=8, timeNs=200), 'except', sync_time_ns=100) == (
        8,
        200,
    )


def test_timestamp_before_sync_is_rejected():
    with pytest.raises(ReceiveStreamError, match='before last sync'):
        validate(StreamResult(ret=8, timeNs=50), 'except', sync_time_ns=100)


def test_missing_timestamp_bypasses_the_sync_check():
    assert validate(StreamResult(ret=8, timeNs=0), 'except', sync_time_ns=100) == (8, 0)


def test_overflow_raises_when_configured(soapy_constants):
    sr = StreamResult(ret=soapy_constants.SOAPY_SDR_OVERFLOW, timeNs=9)
    with pytest.raises(OverflowError, match='overflow'):
        validate(sr, 'except')


@pytest.mark.parametrize('on_overflow', ['ignore', 'log'])
def test_overflow_is_a_zero_length_read_otherwise(soapy_constants, on_overflow):
    sr = StreamResult(ret=soapy_constants.SOAPY_SDR_OVERFLOW, timeNs=9)
    assert validate(sr, on_overflow) == (0, 9)


def test_overflow_still_checks_the_timestamp(soapy_constants):
    sr = StreamResult(ret=soapy_constants.SOAPY_SDR_OVERFLOW, timeNs=50)
    with pytest.raises(ReceiveStreamError, match='before last sync'):
        validate(sr, 'ignore', sync_time_ns=100)


@pytest.mark.parametrize('code', ['TIMEOUT', 'STREAM_ERROR', 'CORRUPTION', 'UNDERFLOW'])
def test_other_errors_name_the_code(soapy_constants, code):
    value = SOAPY_CONSTANTS[f'SOAPY_SDR_{code}']
    with pytest.raises(ReceiveStreamError, match=rf'{code} \(error code {value}\)'):
        validate(StreamResult(ret=value), 'ignore')


# %% compute_overload_info


def _samples(xp, peaks, count=64):
    """one row per port, each with |x| peaking at the given value"""
    rows = []
    for peak in peaks:
        x = np.zeros(count, dtype='complex64')
        x[3] = peak
        x[7] = 0.5 * peak * 1j
        rows.append(x)
    return xp.asarray(np.stack(rows))


def test_no_limits_gives_no_info(xp):
    source = ss.specs.SoapySource(
        master_clock_rate=MCR, adc_overload_limit=None, if_overload_limit=None
    )
    assert soapy.compute_overload_info(_samples(xp, [1.0]), source, _capture()) == {}


def test_adc_headroom_is_floored_relative_to_the_peak(xp):
    # peak 1.0 is -3 dBfs by the module's convention; 0.1 is -23 dBfs
    source = ss.specs.SoapySource(master_clock_rate=MCR, adc_overload_limit=-1)
    capture = _capture(port=(0, 1), gain=(0, -10))
    info = soapy.compute_overload_info(_samples(xp, [1.0, 0.1]), source, capture)
    assert list(info) == ['adc_headroom']
    assert info['adc_headroom'].dtype == xp.int8
    assert sw.array_namespace(info['adc_headroom']) is sw.array_namespace(xp.zeros(1))
    assert info['adc_headroom'].tolist() == [2, 22]


def test_if_headroom_adds_two_thirds_of_the_gain(xp):
    source = ss.specs.SoapySource(
        master_clock_rate=MCR, adc_overload_limit=None, if_overload_limit=-10
    )
    capture = _capture(port=(0, 1), gain=(0, -30))
    info = soapy.compute_overload_info(_samples(xp, [1.0, 1.0]), source, capture)
    assert list(info) == ['if_headroom']
    # floor(-10 - (-3 + 2/3 * gain))
    assert info['if_headroom'].tolist() == [-7, 13]


def test_headroom_is_clipped_to_int8_range(xp):
    source = ss.specs.SoapySource(master_clock_rate=MCR, adc_overload_limit=-1)
    info = soapy.compute_overload_info(_samples(xp, [1e-9]), source, _capture())
    assert info['adc_headroom'].tolist() == [100]


# %% _assign_iq_calibration


@pytest.fixture
def calibration_nc(tmp_path):
    yield save_yfactor_calibration(tmp_path / 'cal.nc')
    # read_calibration and the lookups cache by path
    sw.util.clear_caches()


def _acquired_iq(capture, source):
    return ss.specs.AcquiredIQ(
        pre_align=np.zeros((1, 16), dtype='complex64'),
        pre_filter=None,
        aligned=None,
        capture=capture,
        info=ss.specs.SoapyAcquisitionInfo(start_time=None, backend_sample_rate=MCR),
        extra_data={},
        source_spec=source,
        resampler=design_resampler(capture, MCR),
    )


def test_assign_iq_calibration_scales_voltage_and_adds_system_noise(calibration_nc):
    source = ss.specs.SoapySource(master_clock_rate=MCR, calibration=calibration_nc)
    iq = _acquired_iq(_capture(gain=-10), source)

    soapy._assign_iq_calibration(iq)

    expected_scale = np.sqrt(1 / receiver_gain(-10))
    assert iq.voltage_scale == pytest.approx(expected_scale, rel=1e-6)
    noise = iq.extra_data['system_noise']
    assert noise.dims == ('capture',)
    assert noise.attrs['units'] == 'dBm/Hz'
    assert float(noise) == pytest.approx(5.0 + 10 * np.log10(BOLTZMANN_MW * 290))


def test_assign_iq_calibration_without_a_file_leaves_iq_alone():
    source = ss.specs.SoapySource(master_clock_rate=MCR, calibration=None)
    iq = _acquired_iq(_capture(), source)

    soapy._assign_iq_calibration(iq)

    assert iq.voltage_scale == 1
    assert iq.extra_data == {}


# %% probe_soapy_info (fake device)


@pytest.fixture
def device(fake_soapy):
    return fake_soapy.Device(({},))[0]


def test_probe_soapy_info_fields(device):
    info = soapy.probe_soapy_info(device, retries=3)

    assert isinstance(info, soapy.SoapyInfo)
    assert (info.driver, info.hardware) == ('SoapyAIRT', 'AIR7101B')
    assert (info.num_rx_ports, info.num_tx_ports) == (2, 0)
    assert info.has_timestamps is True
    assert info.timesources == ('internal', 'external', 'gps')
    assert info.registers == ('FPGA',)
    assert info.global_sensors['xcvr_temp'].reading == '41.5'
    assert info.global_sensors['xcvr_temp'].info.units == 'C'
    assert info.retries == 3
    assert info.min_port_count(1) == 2

    port = info.rx_ports[1]
    assert port.full_gain_range == soapy._SoapyRange(minimum=-30, maximum=0, step=0.5)
    assert list(port.gains) == ['PGA']
    assert port.frequencies['RF'][0].maximum == pytest.approx(6e9)
    assert port.master_clock_rates == (125e6,)
    assert port.stream_formats == ('CF32', 'CS16')
    assert port.corrections == ('DC removal',)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason='_probe_channel indexes getSensorInfo(...)[0], but the API returns a '
    'single ArgInfo (as the global-sensor path beside it assumes)',
)
def test_probe_soapy_info_with_a_channel_sensor(device):
    device.channel_sensors['rssi'] = '-40.0'
    info = soapy.probe_soapy_info(device)
    assert info.rx_ports[0].sensors['rssi'].reading == '-40.0'


# %% RxStream (fake device)

RX = 1


def _spec(**kws):
    return ss.specs.SoapySource(master_clock_rate=MCR, **kws)


class StreamAllSpec(ss.specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: ss.specs.types.MasterClockRate = MCR
    rx_enable_delay = 0.35
    stream_all_rx_ports = True


class NoDelaySpec(ss.specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: ss.specs.types.MasterClockRate = MCR
    rx_enable_delay = None


class ComplexTransportSpec(ss.specs.SoapySource, kw_only=True, frozen=True):
    master_clock_rate: ss.specs.types.MasterClockRate = MCR
    transport_dtype = 'complex64'


def _stream(device, spec=None, **kws):
    spec = spec if spec is not None else _spec()
    return soapy.RxStream(spec, soapy.probe_soapy_info(device), **kws)


class TestRxStreamSetup:
    def test_requested_ports_with_gain_minimized(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(1,))

        assert stream.ports == (1,)
        assert device.calls_named('setGain', 'setupStream') == [
            ('setGain', RX, 0, -30.0),
            ('setGain', RX, 1, -30.0),
            ('setupStream', RX, 'CF32', (1,)),
            ('setGain', RX, 0, 0.0),
            ('setGain', RX, 1, 0.0),
        ]

    def test_stream_all_rx_ports_ignores_the_request(self, device):
        stream = _stream(device, StreamAllSpec())
        stream.setup(device, ports=1)
        assert stream.ports == (0, 1)
        assert device.streams[0].channels == (0, 1)

    def test_scalar_port_is_accepted(self, device):
        stream = _stream(device)
        stream.setup(device, ports=0)
        assert stream.ports == (0,)

    def test_no_ports_is_an_error(self, device):
        with pytest.raises(RuntimeError, match='stream_all_rx_ports=False'):
            _stream(device).setup(device)

    def test_int16_transport_selects_cs16(self, device):
        class Int16Spec(ss.specs.SoapySource, kw_only=True, frozen=True):
            master_clock_rate: ss.specs.types.MasterClockRate = MCR
            transport_dtype = 'int16'

        stream = _stream(device, Int16Spec())
        stream.setup(device, ports=(0,))
        assert device.streams[0].format == 'CS16'

    def test_unsupported_transport_is_rejected(self, device):
        with pytest.raises(ValueError, match='unsupported transport type'):
            _stream(device, ComplexTransportSpec()).setup(device, ports=(0,))

    def test_repeated_setup_with_the_same_ports_is_a_no_op(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.setup(device, ports=(0,))
        assert len(device.calls_named('setupStream')) == 1


class TestRxStreamEnable:
    def test_enable_activates_with_a_delayed_timestamp(self, device):
        stream = _stream(device, StreamAllSpec())
        stream.setup(device)
        before = device.getHardwareTime('now')

        stream.enable(device, True)

        ((_, flags, time_ns),) = device.calls_named('activateStream')
        assert flags == 4  # SOAPY_SDR_HAS_TIME
        assert time_ns - before == pytest.approx(0.35e9, abs=0.05e9)
        assert stream.is_enabled
        assert stream.checked_timestamp is False

    def test_enable_without_a_delay_activates_immediately(self, device):
        stream = _stream(device, NoDelaySpec())
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        assert device.calls_named('activateStream') == [('activateStream', 4, 0)]

    def test_enable_is_idempotent_and_disable_deactivates(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))

        stream.enable(device, False)
        stream.enable(device, True)
        stream.enable(device, True)
        stream.enable(device, False)
        stream.enable(device, False)

        assert [
            c[0] for c in device.calls_named('activateStream', 'deactivateStream')
        ] == [
            'activateStream',
            'deactivateStream',
        ]
        assert not stream.is_enabled


def _float_buffers(count, ports=1):
    return [np.zeros(2 * count, dtype='float32') for _ in range(ports)]


class TestRxStreamRead:
    def test_read_fills_from_the_offset(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        (buf,) = _float_buffers(64)

        count, time_ns = stream.read(
            device, [buf], offset=10, count=20, timeout_sec=0.1, last_sync_time=None
        )

        assert count == 20
        assert time_ns == device.streams[0].activate_time_ns
        assert not buf[:20].any()
        assert buf[20:60].all()
        assert not buf[60:].any()
        assert device.calls_named('readStream') == [
            ('readStream', 20, round((0.0 + 0.1 + 0.5) * 1e6))
        ]

    def test_first_read_rejects_a_timestamp_from_before_the_sync(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        stale = device.streams[0].activate_time_ns + 1

        with pytest.raises(ReceiveStreamError, match='before last sync'):
            stream.read(device, _float_buffers(8), 0, 8, 0.1, last_sync_time=stale)

    def test_only_the_first_read_is_checked(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        activation = device.streams[0].activate_time_ns

        stream.read(device, _float_buffers(8), 0, 8, 0.1, last_sync_time=activation)
        stream.read(
            device, _float_buffers(8), 0, 8, 0.1, last_sync_time=activation + 10**12
        )
        assert stream.checked_timestamp

    def test_re_enabling_checks_the_timestamp_again(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        stream.read(device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None)
        stream.enable(device, False)
        stream.enable(device, True)
        assert stream.checked_timestamp is False

    def test_read_uses_the_configured_overflow_policy(self, device, fake_soapy):
        stream = _stream(device, on_overflow='ignore')
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        device.fault_queue = [fake_soapy.SOAPY_SDR_OVERFLOW]

        count, _ = stream.read(
            device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None
        )
        assert count == 0

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason='RxStream.read adds rx_enable_delay to the timeout unguarded, while '
        'enable accepts None',
    )
    def test_read_without_an_enable_delay(self, device):
        stream = _stream(device, NoDelaySpec())
        stream.setup(device, ports=(0,))
        stream.enable(device, True)
        count, _ = stream.read(
            device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None
        )
        assert count == 8


class TestRxStreamClose:
    def test_close_deactivates_and_closes_the_stream(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.enable(device, True)

        stream.close(device)

        assert [
            c[0] for c in device.calls_named('deactivateStream', 'closeStream')
        ] == [
            'deactivateStream',
            'closeStream',
        ]
        assert stream.stream is None
        assert stream.ports == ()
        assert device.streams == []

    def test_close_twice_is_a_no_op(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.close(device)
        stream.close(device)
        assert len(device.calls_named('closeStream')) == 1

    def test_close_tolerates_a_stream_the_device_forgot(self, device):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        device.streams.clear()
        stream.close(device)
        assert stream.stream is None

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="RxStream.close prints 'close stream' to stdout",
    )
    def test_close_is_silent(self, device, capsys):
        stream = _stream(device)
        stream.setup(device, ports=(0,))
        stream.close(device)
        assert capsys.readouterr().out == ''


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='capture_changes_port returns True when the ports are unchanged',
)
def test_capture_changes_port(device):
    stream = _stream(device)
    stream.setup(device, ports=(0, 1))
    assert not stream.capture_changes_port(_capture(port=(0, 1), gain=(0, 0)))
    assert stream.capture_changes_port(_capture(port=0))


# %% HardwareTimeSync (fake device)


class TestHardwareTimeSync:
    def test_host_sync_sets_a_clock_that_is_far_off(self, device):
        sync = soapy.HardwareTimeSync('host')
        before = soapy.time.time()

        sync(device)

        assert isinstance(sync.last_sync_time, int)
        assert sync.last_sync_time / 1e9 == pytest.approx(before, abs=0.05)
        assert device.calls_named('setHardwareTime') == [
            ('setHardwareTime', sync.last_sync_time, 'now')
        ]
        assert device.getHardwareTime('now') / 1e9 == pytest.approx(
            soapy.time.time(), abs=0.05
        )

    def test_host_sync_keeps_a_clock_that_is_close(self, device):
        sync = soapy.HardwareTimeSync('internal')
        sync(device)
        first = sync.last_sync_time

        sync(device)

        assert sync.last_sync_time == first
        assert len(device.calls_named('setHardwareTime')) == 1

    def test_host_sync_needs_hardware_time(self, device, monkeypatch):
        monkeypatch.setattr(device, 'hasHardwareTime', lambda *a: False)
        with pytest.raises(IOError, match='hardware time'):
            soapy.HardwareTimeSync('host').to_host_os(device)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='HardwareTimeSync.__call__ stores the sync time but returns None',
    )
    def test_call_returns_the_sync_time(self, device):
        sync = soapy.HardwareTimeSync('host')
        assert sync(device) == sync.last_sync_time

    @pytest.mark.parametrize(
        'fractional, expected_second',
        [(0.3, 1001), (0.85, 1002)],
        ids=['lagging PPS', 'lagging host'],
    )
    def test_pps_sync_targets_the_next_second(
        self, device, monkeypatch, caplog, fractional, expected_second
    ):
        monkeypatch.setattr(soapy.time, 'time', lambda: 1000 + fractional)
        sync = soapy.HardwareTimeSync('external')

        result = sync.to_external_pps(device)

        assert result == expected_second * 10**9
        assert device.pps_time_set_ns == result
        assert device.pps_count == 2  # one transition was awaited
        assert ('out of sync' in caplog.text) == (fractional > 0.2)

    def test_pps_sync_times_out_without_a_pps_input(
        self, device, fake_soapy, monkeypatch
    ):
        fake_soapy.model.pps_present = False
        clock = iter(range(10))
        monkeypatch.setattr(soapy.time, 'perf_counter', lambda: float(next(clock)))
        monkeypatch.setattr(soapy.time, 'sleep', lambda s: None)

        with pytest.raises(RuntimeError, match='no pps input'):
            soapy.HardwareTimeSync('gps').to_external_pps(device)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='the TypeError message is not an f-string, so the source name is '
        'not interpolated',
    )
    def test_unsupported_source_names_itself(self, device):
        with pytest.raises(TypeError) as exc_info:
            soapy.HardwareTimeSync('bogus')(device)
        assert "'bogus'" in str(exc_info.value)


# %% SoapySource (fake device)


def _source(spec=None, **device_kwargs):
    return soapy.SoapySource(
        spec if spec is not None else StreamAllSpec(), **device_kwargs
    )


class TestSoapySourceOpen:
    def test_requires_the_module(self, monkeypatch):
        monkeypatch.setattr(soapy, 'SoapySDR', None)
        with pytest.raises(ImportError, match='SoapySDR'):
            _source()

    def test_opens_one_device_with_the_kwargs(self, fake_soapy):
        source = _source(driver='SoapyAIRT', serial='1')
        assert source.device is fake_soapy.devices[0]
        assert source.device.kwargs == {'driver': 'SoapyAIRT', 'serial': '1'}
        assert source.get_id() == 'AIR7101B'

    def test_rejects_a_non_sequence_device(self, fake_soapy, monkeypatch):
        monkeypatch.setattr(fake_soapy, 'Device', lambda args: fake_soapy._open({}))
        with pytest.raises(RuntimeError, match='unexpected type'):
            _source()

    def test_get_info_is_probed_once(self, fake_soapy):
        source = _source(StreamAllSpec(receive_retries=2))
        info = source.get_info()
        assert info.retries == 2
        assert source.get_info() is info


class TestSoapySourceSetup:
    def test_host_time_source_configures_the_device(self, fake_soapy):
        source = _source(StreamAllSpec(time_source='host', clock_source='external'))
        source.setup()

        device = fake_soapy.devices[0]
        assert device.calls == [
            ('setGainMode', RX, 0, False),
            ('setGainMode', RX, 1, False),
            ('setTimeSource', 'internal'),
            ('setClockSource', 'external'),
            ('setMasterClockRate', MCR),
        ]
        assert source.rx_stream._on_overflow == 'log'
        assert source.rx_stream.stream is None

    def test_external_time_source_raises_on_overflow(self, fake_soapy):
        source = _source(StreamAllSpec(time_source='external'))
        source.setup()
        assert fake_soapy.devices[0].time_source == 'external'
        assert source.rx_stream._on_overflow == 'except'

    def test_sync_at_open(self, fake_soapy):
        source = _source(StreamAllSpec(time_sync_at='open'))
        source.setup()
        assert len(fake_soapy.devices[0].calls_named('setHardwareTime')) == 1
        assert source.sync_time.last_sync_time is not None

    def test_initial_ports_set_up_the_stream(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(1,))
        assert source.rx_stream.ports == (0, 1)
        assert fake_soapy.devices[0].streams[0].channels == (0, 1)

    @pytest.mark.xfail(
        strict=True,
        raises=RuntimeError,
        reason='SoapySource.setup passes rx_ports to RxStream but then calls '
        'RxStream.setup without them, which only stream_all_rx_ports survives',
    )
    def test_initial_ports_on_a_non_stream_all_source(self, fake_soapy):
        source = _source(_spec())
        source.setup(rx_ports=(1,))
        assert source.rx_stream.ports == (1,)


class TestSoapySourceArm:
    def test_gain_then_frequency_then_sample_rate_per_port(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0, 1))
        device = fake_soapy.devices[0]
        del device.calls[:]

        capture = _capture(port=(0, 1), gain=(0, -10), sample_rate=62.5e6)
        assert source.arm(capture) == capture

        assert device.calls_named('setGain', 'setFrequency', 'setSampleRate') == [
            ('setGain', RX, 0, 0.0),
            ('setFrequency', RX, 0, 1e9),
            ('setSampleRate', RX, 0, 62.5e6),
            ('setGain', RX, 1, -10.0),
            ('setFrequency', RX, 1, 1e9),
            ('setSampleRate', RX, 1, 62.5e6),
        ]

    def test_host_resampling_tunes_to_the_designed_backend_rate(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        capture = _capture(sample_rate=50e6, host_resample=True)
        source.arm(capture)
        fs_sdr = design_resampler(capture, MCR)['fs_sdr']
        assert fs_sdr == pytest.approx(62.5e6)
        assert fake_soapy.devices[0].sample_rate == fs_sdr

    def test_external_lo_tunes_to_the_difference_frequency(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        source.arm(_capture(center_frequency=7.35e9, external_lo_frequency=11.38e9))
        assert fake_soapy.devices[0].frequencies[RX, 0] == pytest.approx(4.03e9)

    def test_arm_disables_a_running_stream(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        source.arm(_capture())
        source.trigger()
        source.arm(_capture(gain=-5))
        assert not source.rx_stream.is_enabled

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='capture_changes_port is inverted and SoapySource.arm never records '
        '_capture, so a port change on a non-stream_all source does not rebuild '
        'the stream',
    )
    def test_port_change_rebuilds_the_stream(self, fake_soapy):
        source = _source(_spec())
        source.setup()
        source.arm(_capture(port=0))
        source.arm(_capture(port=1))
        assert source.rx_stream.ports == (1,)
        assert fake_soapy.devices[0].streams[-1].channels == (1,)


class TestSoapySourceAcquire:
    def _acquire(self, source, capture, count):
        source.arm(capture)
        source.trigger()
        ports = len(ss.specs.helpers.split_capture_ports(capture))
        samples = np.zeros((ports, count), dtype='complex64')
        bufs = [row.view('float32') for row in samples]
        received, time_ns = source.read(bufs, 0, count, 0.01)
        return samples, received, time_ns

    def test_trigger_syncs_and_enables(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0, 1))
        device = fake_soapy.devices[0]

        samples, received, time_ns = self._acquire(
            source, _capture(port=(0, 1), gain=(0, 0)), 4096
        )

        assert received == 4096
        assert time_ns == device.streams[0].activate_time_ns
        assert time_ns >= source.sync_time.last_sync_time
        assert samples.all()
        assert [
            c[0] for c in device.calls_named('setHardwareTime', 'activateStream')
        ] == [
            'setHardwareTime',
            'activateStream',
        ]

    def test_retrigger_deactivates_first(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        self._acquire(source, _capture(), 64)
        source.prepare_retrigger()
        source.trigger()
        device = fake_soapy.devices[0]
        names = [c[0] for c in device.calls_named('activateStream', 'deactivateStream')]
        assert names == ['activateStream', 'deactivateStream', 'activateStream']

    def test_package_iq(self, fake_soapy):
        source = _source(StreamAllSpec(adc_overload_limit=-1))
        source.setup(rx_ports=(0,))
        capture = _capture(center_frequency=7.35e9, external_lo_frequency=11.38e9)
        samples, _, time_ns = self._acquire(source, capture, 4096)
        iq = ss.specs.AcquiredIQ(
            pre_align=samples,
            pre_filter=None,
            aligned=None,
            capture=capture,
            info=ss.specs.AcquisitionInfo(source_id='x'),
            extra_data={},
            source_spec=source.spec,
            resampler=source.get_resampler(capture),
        )

        iq = source.package_iq(iq, samples, time_ns)

        assert isinstance(iq.info, ss.specs.SoapyAcquisitionInfo)
        assert iq.info.start_time.value == time_ns
        assert iq.info.backend_sample_rate == MCR
        assert iq.info.source_id == 'x'
        assert iq.conjugate == (True,)  # high-side LO
        assert iq.extra_data['adc_headroom'].dtype == np.int8

    def test_package_iq_without_a_timestamp(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        capture = _capture()
        samples, _, _ = self._acquire(source, capture, 64)
        iq = ss.specs.AcquiredIQ(
            pre_align=samples,
            pre_filter=None,
            aligned=None,
            capture=capture,
            info=ss.specs.AcquisitionInfo(),
            extra_data={},
            source_spec=source.spec,
            resampler=source.get_resampler(capture),
        )
        iq = source.package_iq(iq, samples, None)
        assert iq.info.start_time is None
        assert iq.conjugate == (False,)


class TestSoapySourceClose:
    def test_close_survives_a_torn_down_module(self, fake_soapy):
        source = _source()
        source.setup()
        fake_soapy._SoapySDR.Device_deactivateStream = None
        source.close()

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='SoapySource.close reads self._device and self._rx_stream, which do '
        'not exist (the attributes are device and rx_stream), so nothing is closed',
    )
    def test_close_releases_the_stream_and_device(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        source.close()
        device = fake_soapy.devices[0]
        assert device.closed
        assert device.streams == []


# %% Controller.acquire through the fake_soapy binding


def _controller(fake_soapy_ext, **spec_kws):
    ctrl_cls = ss.lib.bindings.get_controller('fake_soapy')
    spec = fake_soapy_ext.FakeSoapySourceSpec(**spec_kws)
    return ctrl_cls.from_source_spec(spec, rx_ports=(0, 1))


def _dBfs(x):
    return 10 * np.log10(np.mean(np.abs(np.asarray(x)) ** 2, axis=-1))


class TestControllerAcquire:
    def test_start_time_and_sample_count(self, fake_soapy, fake_soapy_ext):
        capture = _capture(port=(0, 1), gain=(0, 0), duration=2e-3)
        with _controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(capture)
            overlaps = ss.lib.compute.get_correction_overlaps(capture, ctrl.source_spec)
            iq = ctrl.acquire()

            device = fake_soapy.devices[0]
            stream = device.streams[0]
            holdoff = round(2e-3 * MCR)
            assert iq.pre_align.shape == (2, round(2e-3 * MCR) + sum(overlaps))
            assert iq.pre_align.dtype == np.complex64
            expected_start = stream.activate_time_ns + round(
                (holdoff + overlaps[0]) * 1e9 / MCR
            )
            assert iq.info.start_time.value == expected_start
            assert iq.info.backend_sample_rate == MCR
            assert [c[1] for c in device.calls_named('readStream')] == [
                iq.pre_align.shape[1],
                holdoff,
            ]

    def test_single_port_capture_streams_both_and_keeps_its_own(
        self, fake_soapy, fake_soapy_ext
    ):
        fake_soapy.model.tones[1] = (1.505e9, -60.0)
        with _controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(_capture(port=1, duration=1e-3))
            iq = ctrl.acquire()
            assert fake_soapy.devices[0].streams[0].channels == (0, 1)

        # -60 dBm through 50 dB of gain is -10 dBfs; port 0 would be noise only
        assert iq.pre_align.shape[0] == 1
        assert _dBfs(iq.pre_align)[0] == pytest.approx(-10.0, abs=0.05)

    def test_noise_level_matches_the_model(self, fake_soapy, fake_soapy_ext):
        with _controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(_capture(port=(0, 1), gain=(0, -10), duration=1e-3))
            iq = ctrl.acquire()

        expected = [
            10 * np.log10(fake_soapy.model.noise_variance(0, 0, MCR, 1e9)),
            10 * np.log10(fake_soapy.model.noise_variance(1, -10, MCR, 1e9)),
        ]
        assert _dBfs(iq.pre_align) == pytest.approx(expected, abs=0.05)

    def test_stale_first_timestamp_is_a_stream_error(self, fake_soapy, fake_soapy_ext):
        with _controller(fake_soapy_ext, receive_retries=0) as ctrl:
            ctrl._arm_spec(_capture(duration=1e-3))
            device = fake_soapy.devices[0]
            activate = device.activateStream

            def activate_in_the_past(stream, **kws):
                activate(stream, **kws)
                stream.activate_time_ns = 1

            device.activateStream = activate_in_the_past
            with pytest.raises(ReceiveStreamError, match='before last sync'):
                ctrl.acquire()
