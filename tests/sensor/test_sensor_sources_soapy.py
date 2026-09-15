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
