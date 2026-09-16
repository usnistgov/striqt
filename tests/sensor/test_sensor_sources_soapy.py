"""striqt.sensor.lib.sources.soapy: the pieces that run without a SoapySDR module.
Stream reads, overload accounting, capability structs, time source mapping, and
calibration assignment onto acquired IQ"""

from __future__ import annotations

import msgspec
import numpy as np
import pytest
from fake_soapy import CONSTANTS, ERROR_NAMES, RX, ArgInfo, Range, StreamResult
from soapy_factories import MCR, call_names, fake_controller, soapy_capture, source_spec
from sweep_strategies import BOLTZMANN_MW, receiver_gain

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.compute import design_resampler
from striqt.sensor.lib.sources import soapy
from striqt.sensor.lib.sources.base import ReceiveStreamError

OVERFLOW = CONSTANTS['SOAPY_SDR_OVERFLOW']


# %% _SoapyRange and _SoapyArgInfo


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='_SoapyArgInfo.range is annotated as a bare tuple, so validate() leaves '
    'the _SoapyRange entries as dicts',
)
def test_arg_info_validate_round_trips():
    info = soapy._SoapyArgInfo.from_soapy(
        ArgInfo('gain', units='dB', value='0', options=('a', 'b'))
    )
    assert info.validate() == info


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
    assert soapy.device_time_source(source_spec(time_source=time_source)) == expected


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


@pytest.mark.parametrize('on_overflow', ['ignore', 'log'])
def test_overflow_is_a_zero_length_read_otherwise(fake_soapy, on_overflow):
    assert validate(StreamResult(ret=OVERFLOW, timeNs=9), on_overflow) == (0, 9)


def _named_error(name):
    value = CONSTANTS[f'SOAPY_SDR_{name}']
    return pytest.param(
        StreamResult(ret=value),
        'ignore',
        None,
        ReceiveStreamError,
        rf'{name} \(error code {value}\)',
        id=name.lower(),
    )


@pytest.mark.parametrize(
    'result, on_overflow, sync_time_ns, exc, match',
    [
        pytest.param(
            StreamResult(ret=8, timeNs=50),
            'except',
            100,
            ReceiveStreamError,
            'before last sync',
            id='timestamp-before-sync',
        ),
        pytest.param(
            StreamResult(ret=OVERFLOW, timeNs=9),
            'except',
            None,
            OverflowError,
            'overflow',
            id='overflow-except',
        ),
        pytest.param(
            StreamResult(ret=OVERFLOW, timeNs=50),
            'ignore',
            100,
            ReceiveStreamError,
            'before last sync',
            id='overflow-before-sync',
        ),
        _named_error('TIMEOUT'),
        _named_error('STREAM_ERROR'),
        _named_error('CORRUPTION'),
        _named_error('UNDERFLOW'),
    ],
)
def test_validate_stream_read_errors(
    fake_soapy, result, on_overflow, sync_time_ns, exc, match
):
    with pytest.raises(exc, match=match):
        validate(result, on_overflow, sync_time_ns=sync_time_ns)


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
    source = source_spec(adc_overload_limit=None, if_overload_limit=None)
    capture = soapy_capture()
    assert soapy.compute_overload_info(_samples(xp, [1.0]), source, capture) == {}


def test_adc_headroom_has_one_entry_per_port(xp):
    source = source_spec(adc_overload_limit=-1)
    capture = soapy_capture(port=(0, 1), gain=(0, -10))
    info = soapy.compute_overload_info(_samples(xp, [1.0, 0.1]), source, capture)
    assert list(info) == ['adc_headroom']
    headroom = info['adc_headroom']
    assert headroom.dtype == xp.int8
    assert headroom.shape == (2,)
    assert sw.array_namespace(headroom) is sw.array_namespace(xp.zeros(1))
    # the port with the smaller peak has more room before the limit
    assert headroom[1] > headroom[0]


def test_if_headroom_needs_the_if_limit(xp):
    source = source_spec(adc_overload_limit=None, if_overload_limit=-10)
    capture = soapy_capture(port=(0, 1), gain=(0, -30))
    info = soapy.compute_overload_info(_samples(xp, [1.0, 1.0]), source, capture)
    assert list(info) == ['if_headroom']
    assert info['if_headroom'].dtype == xp.int8
    assert info['if_headroom'].shape == (2,)


# %% _assign_iq_calibration


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
    source = source_spec(calibration=calibration_nc)
    iq = _acquired_iq(soapy_capture(gain=-10), source)

    soapy._assign_iq_calibration(iq)

    expected_scale = np.sqrt(1 / receiver_gain(-10))
    assert iq.voltage_scale == pytest.approx(expected_scale, rel=1e-6)
    noise = iq.extra_data['system_noise']
    assert noise.dims == ('capture',)
    assert noise.attrs['units'] == 'dBm/Hz'
    assert noise.item() == pytest.approx(5.0 + 10 * np.log10(BOLTZMANN_MW * 290))


def test_assign_iq_calibration_without_a_file_leaves_iq_alone():
    iq = _acquired_iq(soapy_capture(), source_spec(calibration=None))

    soapy._assign_iq_calibration(iq)

    assert iq.voltage_scale == 1
    assert iq.extra_data == {}


# %% probe_soapy_info (fake device)


def test_probe_soapy_info_fields(soapy_device):
    info = soapy.probe_soapy_info(soapy_device, retries=3)

    assert isinstance(info, soapy.SoapyInfo)
    assert (info.driver, info.hardware) == ('SoapyAIRT', 'AIR7101B')
    assert info.hardware_info == {'firmware': '1.0.0', 'fpga': 'fake'}
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
    assert port.frequencies['RF'][0].maximum == pytest.approx(6e9)
    assert port.master_clock_rates == (125e6,)
    assert port.stream_formats == ('CF32', 'CS16')


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason='_probe_channel indexes getSensorInfo(...)[0], but the API returns a '
    'single ArgInfo (as the global-sensor path beside it assumes)',
)
def test_probe_soapy_info_with_a_channel_sensor(soapy_device):
    soapy_device.channel_sensors['rssi'] = '-40.0'
    info = soapy.probe_soapy_info(soapy_device)
    assert info.rx_ports[0].sensors['rssi'].reading == '-40.0'


# %% RxStream (fake device)


def _stream(device, spec=None, **kws):
    """an RxStream on `device` that is not yet set up; `soapy_stream` covers the
    default spec on port 0"""
    if spec is None:
        spec = source_spec()
    return soapy.RxStream(spec, soapy.probe_soapy_info(device), **kws)


class TestRxStreamSetup:
    def test_requested_ports_with_gain_minimized(self, soapy_device):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=(1,))

        assert stream.ports == (1,)
        assert soapy_device.calls_named('setGain', 'setupStream') == [
            ('setGain', RX, 0, -30.0),
            ('setGain', RX, 1, -30.0),
            ('setupStream', RX, 'CF32', (1,)),
            ('setGain', RX, 0, 0.0),
            ('setGain', RX, 1, 0.0),
        ]

    def test_stream_all_rx_ports_ignores_the_request(self, soapy_device):
        stream = _stream(soapy_device, source_spec(stream_all_rx_ports=True))
        stream.setup(soapy_device, ports=1)
        assert stream.ports == (0, 1)
        assert soapy_device.streams[0].channels == (0, 1)

    def test_scalar_port_is_accepted(self, soapy_device):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=0)
        assert stream.ports == (0,)

    def test_no_ports_is_an_error(self, soapy_device):
        with pytest.raises(RuntimeError, match='stream_all_rx_ports=False'):
            _stream(soapy_device).setup(soapy_device)

    def test_int16_transport_selects_cs16(self, soapy_device):
        stream = _stream(soapy_device, source_spec(transport_dtype='int16'))
        stream.setup(soapy_device, ports=(0,))
        assert soapy_device.streams[0].format == 'CS16'

    def test_unsupported_transport_is_rejected(self, soapy_device):
        stream = _stream(soapy_device, source_spec(transport_dtype='complex64'))
        with pytest.raises(ValueError, match='unsupported transport type'):
            stream.setup(soapy_device, ports=(0,))

    def test_repeated_setup_with_the_same_ports_is_a_no_op(
        self, soapy_device, soapy_stream
    ):
        soapy_stream.setup(soapy_device, ports=(0,))
        assert len(soapy_device.calls_named('setupStream')) == 1


class TestRxStreamEnable:
    def test_enable_activates_with_a_delayed_timestamp(self, soapy_device):
        spec = source_spec(stream_all_rx_ports=True, rx_enable_delay=0.35)
        stream = _stream(soapy_device, spec)
        stream.setup(soapy_device)
        before = soapy_device.getHardwareTime('now')

        stream.enable(soapy_device, True)

        ((_, flags, time_ns),) = soapy_device.calls_named('activateStream')
        assert flags == 4  # SOAPY_SDR_HAS_TIME
        assert time_ns - before == pytest.approx(0.35e9, abs=0.05e9)
        assert stream.is_enabled
        assert stream.checked_timestamp is False

    def test_enable_without_a_delay_activates_immediately(self, soapy_device):
        stream = _stream(soapy_device, source_spec(rx_enable_delay=None))
        stream.setup(soapy_device, ports=(0,))
        stream.enable(soapy_device, True)
        assert soapy_device.calls_named('activateStream') == [('activateStream', 4, 0)]

    def test_enable_is_idempotent_and_disable_deactivates(self, soapy_device):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=(0,))

        stream.enable(soapy_device, False)
        stream.enable(soapy_device, True)
        stream.enable(soapy_device, True)
        stream.enable(soapy_device, False)
        stream.enable(soapy_device, False)

        assert call_names(soapy_device, 'activateStream', 'deactivateStream') == [
            'activateStream',
            'deactivateStream',
        ]
        assert not stream.is_enabled


def _float_buffers(count, ports=1):
    return [np.zeros(2 * count, dtype='float32') for _ in range(ports)]


class TestRxStreamRead:
    def test_read_fills_from_the_offset(self, soapy_device, soapy_stream):
        (buf,) = _float_buffers(64)

        count, time_ns = soapy_stream.read(
            soapy_device,
            [buf],
            offset=10,
            count=20,
            timeout_sec=0.1,
            last_sync_time=None,
        )

        assert count == 20
        assert time_ns == soapy_device.streams[0].activate_time_ns
        assert not buf[:20].any()
        assert buf[20:60].all()
        assert not buf[60:].any()
        assert soapy_device.calls_named('readStream') == [
            ('readStream', 20, round((0.0 + 0.1 + 0.5) * 1e6))
        ]

    def test_first_read_rejects_a_timestamp_from_before_the_sync(
        self, soapy_device, soapy_stream
    ):
        stale = soapy_device.streams[0].activate_time_ns + 1

        with pytest.raises(ReceiveStreamError, match='before last sync'):
            soapy_stream.read(
                soapy_device, _float_buffers(8), 0, 8, 0.1, last_sync_time=stale
            )

    def test_only_the_first_read_is_checked(self, soapy_device, soapy_stream):
        activation = soapy_device.streams[0].activate_time_ns

        soapy_stream.read(
            soapy_device, _float_buffers(8), 0, 8, 0.1, last_sync_time=activation
        )
        soapy_stream.read(
            soapy_device,
            _float_buffers(8),
            0,
            8,
            0.1,
            last_sync_time=activation + 10**12,
        )
        assert soapy_stream.checked_timestamp

    def test_re_enabling_checks_the_timestamp_again(self, soapy_device, soapy_stream):
        soapy_stream.read(
            soapy_device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None
        )
        soapy_stream.enable(soapy_device, False)
        soapy_stream.enable(soapy_device, True)
        assert soapy_stream.checked_timestamp is False

    def test_read_uses_the_configured_overflow_policy(self, soapy_device):
        stream = _stream(soapy_device, on_overflow='ignore')
        stream.setup(soapy_device, ports=(0,))
        stream.enable(soapy_device, True)
        soapy_device.fault_queue = [OVERFLOW]

        count, _ = stream.read(
            soapy_device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None
        )
        assert count == 0

    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason='RxStream.read adds rx_enable_delay to the timeout unguarded, while '
        'enable accepts None',
    )
    def test_read_without_an_enable_delay(self, soapy_device):
        stream = _stream(soapy_device, source_spec(rx_enable_delay=None))
        stream.setup(soapy_device, ports=(0,))
        stream.enable(soapy_device, True)
        count, _ = stream.read(
            soapy_device, _float_buffers(8), 0, 8, 0.1, last_sync_time=None
        )
        assert count == 8


class TestRxStreamClose:
    def test_close_deactivates_and_closes_the_stream(self, soapy_device, soapy_stream):
        soapy_stream.close(soapy_device)

        assert call_names(soapy_device, 'deactivateStream', 'closeStream') == [
            'deactivateStream',
            'closeStream',
        ]
        assert soapy_stream.stream is None
        assert soapy_stream.ports == ()
        assert soapy_device.streams == []

    def test_close_twice_is_a_no_op(self, soapy_device):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=(0,))
        stream.close(soapy_device)
        stream.close(soapy_device)
        assert len(soapy_device.calls_named('closeStream')) == 1

    def test_close_tolerates_a_stream_the_device_forgot(self, soapy_device):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=(0,))
        soapy_device.streams.clear()
        stream.close(soapy_device)
        assert stream.stream is None

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="RxStream.close prints 'close stream' to stdout",
    )
    def test_close_is_silent(self, soapy_device, capsys):
        stream = _stream(soapy_device)
        stream.setup(soapy_device, ports=(0,))
        stream.close(soapy_device)
        assert capsys.readouterr().out == ''


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason='capture_changes_port returns True when the ports are unchanged',
)
def test_capture_changes_port(soapy_device):
    stream = _stream(soapy_device)
    stream.setup(soapy_device, ports=(0, 1))
    assert not stream.capture_changes_port(soapy_capture(port=(0, 1), gain=(0, 0)))
    assert stream.capture_changes_port(soapy_capture(port=0))


# %% HardwareTimeSync (fake device)


class TestHardwareTimeSync:
    def test_host_sync_sets_a_clock_that_is_far_off(self, soapy_device):
        sync = soapy.HardwareTimeSync('host')
        before = soapy.time.time()

        sync(soapy_device)

        assert isinstance(sync.last_sync_time, int)
        assert sync.last_sync_time / 1e9 == pytest.approx(before, abs=0.05)
        assert soapy_device.calls_named('setHardwareTime') == [
            ('setHardwareTime', sync.last_sync_time, 'now')
        ]
        assert soapy_device.getHardwareTime('now') / 1e9 == pytest.approx(
            soapy.time.time(), abs=0.05
        )

    def test_host_sync_needs_hardware_time(self, soapy_device, monkeypatch):
        monkeypatch.setattr(soapy_device, 'hasHardwareTime', lambda *a: False)
        with pytest.raises(IOError, match='hardware time'):
            soapy.HardwareTimeSync('host').to_host_os(soapy_device)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='HardwareTimeSync.__call__ stores the sync time but returns None',
    )
    def test_call_returns_the_sync_time(self, soapy_device):
        sync = soapy.HardwareTimeSync('host')
        assert sync(soapy_device) == sync.last_sync_time

    def test_pps_sync_applies_a_whole_second_after_a_transition(
        self, soapy_device, monkeypatch
    ):
        monkeypatch.setattr(soapy.time, 'time', lambda: 1000.3)
        sync = soapy.HardwareTimeSync('external')

        result = sync.to_external_pps(soapy_device)

        assert result % 10**9 == 0
        assert result > 1000.3e9
        assert soapy_device.pps_time_set_ns == result
        assert soapy_device.pps_count == 2  # one transition was awaited

    def test_pps_sync_times_out_without_a_pps_input(
        self, soapy_device, fake_soapy, monkeypatch
    ):
        fake_soapy.model.pps_present = False
        clock = iter(range(10))
        monkeypatch.setattr(soapy.time, 'perf_counter', lambda: float(next(clock)))
        monkeypatch.setattr(soapy.time, 'sleep', lambda s: None)

        with pytest.raises(RuntimeError, match='no pps input'):
            soapy.HardwareTimeSync('gps').to_external_pps(soapy_device)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='the TypeError message is not an f-string, so the source name is '
        'not interpolated',
    )
    def test_unsupported_source_names_itself(self, soapy_device):
        with pytest.raises(TypeError) as exc_info:
            soapy.HardwareTimeSync('bogus')(soapy_device)
        assert "'bogus'" in str(exc_info.value)


# %% SoapySource (fake device)

STREAM_ALL = {'stream_all_rx_ports': True, 'rx_enable_delay': 0.35}


def _source(**spec_kws):
    """a SoapySource that streams every rx port after an enable delay"""
    return soapy.SoapySource(source_spec(**STREAM_ALL, **spec_kws))


class TestSoapySourceOpen:
    def test_requires_the_module(self, monkeypatch):
        monkeypatch.setattr(soapy, 'SoapySDR', None)
        with pytest.raises(ImportError, match='SoapySDR'):
            _source()

    def test_opens_one_device_with_the_kwargs(self, fake_soapy):
        source = soapy.SoapySource(
            source_spec(**STREAM_ALL), driver='SoapyAIRT', serial='1'
        )
        assert source.device is fake_soapy.devices[0]
        assert source.device.kwargs == {'driver': 'SoapyAIRT', 'serial': '1'}
        assert source.get_id() == 'AIR7101B'

    def test_rejects_a_non_sequence_device(self, fake_soapy, monkeypatch):
        monkeypatch.setattr(fake_soapy, 'Device', lambda args: fake_soapy._open({}))
        with pytest.raises(RuntimeError, match='unexpected type'):
            _source()

    def test_get_info_is_probed_once(self, fake_soapy):
        source = _source(receive_retries=2)
        info = source.get_info()
        assert info.retries == 2
        assert source.get_info() is info


class TestSoapySourceSetup:
    def test_host_time_source_configures_the_device(self, fake_soapy):
        source = _source(time_source='host', clock_source='external')
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
        source = _source(time_source='external')
        source.setup()
        assert fake_soapy.devices[0].time_source == 'external'
        assert source.rx_stream._on_overflow == 'except'

    def test_sync_at_open(self, fake_soapy):
        source = _source(time_sync_at='open')
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
        source = soapy.SoapySource(source_spec())
        source.setup(rx_ports=(1,))
        assert source.rx_stream.ports == (1,)


class TestSoapySourceArm:
    def test_gain_is_set_before_frequency_per_port(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0, 1))
        device = fake_soapy.devices[0]
        del device.calls[:]

        capture = soapy_capture(port=(0, 1), gain=(0, -10), sample_rate=62.5e6)
        assert source.arm(capture) == capture

        # gain before frequency, so the attenuator settles during the retune
        assert device.calls_named('setGain', 'setFrequency') == [
            ('setGain', RX, 0, 0.0),
            ('setFrequency', RX, 0, 1e9),
            ('setGain', RX, 1, -10.0),
            ('setFrequency', RX, 1, 1e9),
        ]
        assert sorted(device.calls_named('setSampleRate')) == [
            ('setSampleRate', RX, 0, 62.5e6),
            ('setSampleRate', RX, 1, 62.5e6),
        ]

    def test_host_resampling_tunes_to_the_designed_backend_rate(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        capture = soapy_capture(sample_rate=50e6, host_resample=True)
        source.arm(capture)
        fs_sdr = design_resampler(capture, MCR)['fs_sdr']
        assert fs_sdr == pytest.approx(62.5e6)
        assert fake_soapy.devices[0].sample_rate == fs_sdr

    def test_external_lo_tunes_to_the_difference_frequency(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        source.arm(
            soapy_capture(center_frequency=7.35e9, external_lo_frequency=11.38e9)
        )
        assert fake_soapy.devices[0].frequencies[RX, 0] == pytest.approx(4.03e9)

    def test_arm_disables_a_running_stream(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        source.arm(soapy_capture())
        source.trigger()
        source.arm(soapy_capture(gain=-5))
        assert not source.rx_stream.is_enabled

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason='capture_changes_port is inverted and SoapySource.arm never records '
        '_capture, so a port change on a non-stream_all source does not rebuild '
        'the stream',
    )
    def test_port_change_rebuilds_the_stream(self, fake_soapy):
        source = soapy.SoapySource(source_spec())
        source.setup()
        source.arm(soapy_capture(port=0))
        source.arm(soapy_capture(port=1))
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
            source, soapy_capture(port=(0, 1), gain=(0, 0)), 4096
        )

        assert received == 4096
        assert time_ns == device.streams[0].activate_time_ns
        assert time_ns >= source.sync_time.last_sync_time
        assert samples.all()
        assert call_names(device, 'setHardwareTime', 'activateStream') == [
            'setHardwareTime',
            'activateStream',
        ]

    def test_retrigger_deactivates_first(self, fake_soapy):
        source = _source()
        source.setup(rx_ports=(0,))
        self._acquire(source, soapy_capture(), 64)
        source.prepare_retrigger()
        source.trigger()
        names = call_names(fake_soapy.devices[0], 'activateStream', 'deactivateStream')
        assert names == ['activateStream', 'deactivateStream', 'activateStream']

    def test_package_iq(self, fake_soapy):
        source = _source(adc_overload_limit=-1)
        source.setup(rx_ports=(0,))
        capture = soapy_capture(center_frequency=7.35e9, external_lo_frequency=11.38e9)
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
        capture = soapy_capture()
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


def _dBfs(x):
    return 10 * np.log10(np.mean(np.abs(np.asarray(x)) ** 2, axis=-1))


class TestControllerAcquire:
    def test_start_time_and_sample_count(self, fake_soapy, fake_soapy_ext):
        capture = soapy_capture(port=(0, 1), gain=(0, 0), duration=2e-3)
        with fake_controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(capture)
            overlaps = ss.lib.compute.get_correction_overlaps(capture, ctrl.source_spec)
            iq = ctrl.acquire()

            device = fake_soapy.devices[0]
            stream = device.streams[0]
            holdoff = round(2e-3 * MCR)
            assert iq.pre_align.shape == (2, round(2e-3 * MCR) + sum(overlaps))
            assert iq.pre_align.dtype == np.complex64
            assert iq.info.backend_sample_rate == MCR
            read_sizes = [c[1] for c in device.calls_named('readStream')]
            assert read_sizes == [iq.pre_align.shape[1], holdoff]

            # the timestamp lies inside the acquisition window
            window_ns = round(sum(read_sizes) * 1e9 / MCR)
            assert stream.activate_time_ns <= iq.info.start_time.value
            assert iq.info.start_time.value <= stream.activate_time_ns + window_ns

    def test_single_port_capture_streams_both_and_keeps_its_own(
        self, fake_soapy, fake_soapy_ext
    ):
        fake_soapy.model.tones[1] = (1.505e9, -60.0)
        with fake_controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(soapy_capture(port=1, duration=1e-3))
            iq = ctrl.acquire()
            assert fake_soapy.devices[0].streams[0].channels == (0, 1)

        # -60 dBm through 50 dB of gain is -10 dBfs; port 0 would be noise only
        assert iq.pre_align.shape[0] == 1
        assert _dBfs(iq.pre_align)[0] == pytest.approx(-10.0, abs=0.05)

    def test_noise_level_matches_the_model(self, fake_soapy, fake_soapy_ext):
        with fake_controller(fake_soapy_ext) as ctrl:
            ctrl._arm_spec(soapy_capture(port=(0, 1), gain=(0, -10), duration=1e-3))
            iq = ctrl.acquire()

        expected = [
            10 * np.log10(fake_soapy.model.noise_variance(0, 0, MCR, 1e9)),
            10 * np.log10(fake_soapy.model.noise_variance(1, -10, MCR, 1e9)),
        ]
        assert _dBfs(iq.pre_align) == pytest.approx(expected, abs=0.05)

    def test_stale_first_timestamp_is_a_stream_error(self, fake_soapy, fake_soapy_ext):
        with fake_controller(fake_soapy_ext, receive_retries=0) as ctrl:
            ctrl._arm_spec(soapy_capture(duration=1e-3))
            device = fake_soapy.devices[0]
            activate = device.activateStream

            def activate_in_the_past(stream, **kws):
                activate(stream, **kws)
                stream.activate_time_ns = 1

            device.activateStream = activate_in_the_past
            with pytest.raises(ReceiveStreamError, match='before last sync'):
                ctrl.acquire()


# %% real SoapySDR bindings: the null driver (pixi environments)


@pytest.fixture(scope='module')
def real_soapy():
    """the real SoapySDR module, as striqt imported it; skipped under uv"""
    pytest.importorskip('SoapySDR')
    if (
        soapy.SoapySDR is None
        or getattr(soapy.SoapySDR, '__name__', None) != 'SoapySDR'
    ):
        pytest.skip('striqt did not import SoapySDR at load time')
    return soapy.SoapySDR


@pytest.fixture
def null_device(real_soapy):
    devices = real_soapy.Device(({'driver': 'null'},))
    assert isinstance(devices, (list, tuple)) and len(devices) == 1
    return devices[0]


class TestNullDriver:
    def test_device_sequence_form(self, real_soapy, null_device):
        # the SWIG proxy is not an instance of the module's Device class
        assert type(null_device).__name__ == 'Device'
        assert null_device.getDriverKey() == 'null'
        assert real_soapy.Device.enumerate({'driver': 'null'}) == ()

    def test_fake_constants_match_the_library(self, real_soapy, subtests):
        for name, value in CONSTANTS.items():
            with subtests.test(msg=name):
                assert getattr(real_soapy, name) == value
        for value, name in ERROR_NAMES.items():
            with subtests.test(msg=f'errToStr({value})'):
                assert real_soapy.errToStr(value) == name

    def test_fake_types_match_the_library(self, real_soapy, subtests):
        info = real_soapy.ArgInfo()
        for attr in vars(ArgInfo('x')):
            with subtests.test(msg=f'ArgInfo.{attr}'):
                assert hasattr(info, attr)
        with pytest.raises(TypeError):
            info[0]

        span = real_soapy.Range(1.0, 2.0, 0.5)
        fake_span = Range(1.0, 2.0, 0.5)
        assert (span.minimum(), span.maximum(), span.step()) == (
            fake_span.minimum(),
            fake_span.maximum(),
            fake_span.step(),
        )
        assert real_soapy.Range(1.0, 2.0).step() == pytest.approx(0.0)

        result = real_soapy.StreamResult()
        assert (result.ret, result.flags, result.timeNs, result.chanMask) == (
            0,
            0,
            0,
            0,
        )

    def test_probe_null_device(self, null_device):
        info = soapy.probe_soapy_info(null_device)
        assert (info.driver, info.hardware) == ('null', 'null')
        assert (info.num_rx_ports, info.num_tx_ports) == (0, 0)
        assert info.has_timestamps is False
        assert info.rx_ports == ()
        assert info.timesources == ()
        assert info.min_port_count(2) == 0

    def test_probe_null_device_round_trips(self, null_device):
        info = soapy.probe_soapy_info(null_device)
        assert dict(info.hardware_info) == {}
        assert info.validate() == info

    def test_time_sync_needs_hardware_time(self, null_device):
        with pytest.raises(IOError, match='hardware time'):
            soapy.HardwareTimeSync('host').to_host_os(null_device)
        with pytest.raises(IOError, match='hardware time'):
            soapy.HardwareTimeSync('external').to_external_pps(null_device)

    def test_soapy_source_opens_and_closes(self, real_soapy):
        source = soapy.SoapySource(source_spec(), driver='null')
        assert source.get_id() == 'null'
        assert source.get_info().num_rx_ports == 0
        source.setup()
        assert source.rx_stream.stream is None
        source.close()


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
def airstack_source_spec(real_soapy):
    if not real_soapy.Device.enumerate({'driver': 'SoapyAIRT'}):
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
