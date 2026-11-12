"""striqt.sensor.lib.compute.summary: the per-port overload state derived from the
headroom that a SoapySDR source reports, the spectrogram SNR above the calibrated
noise floor, and the text of the capture, acquisition, and analysis fields"""

from __future__ import annotations

import math

import pytest
from soapy_factories import soapy_capture
from sweep_strategies import (
    BOLTZMANN_MW,
    T_REF,
    air_sweep,
    make_delayed,
    spectrogram_result,
)
from synthetic_sources import (
    ANALYSIS,
    SCALE_ONLY,
    FakeTrigger,
    preset_capture,
    preset_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib.compute import summary

AIR = ss.bindings.air7101b

# the capture under test spans 2 ports at 1 GHz and 2 GHz, and the sweep also
# visits 2 GHz on port 0 at 30 dB more gain
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)
HIGH_GAIN = soapy_capture(port=0, center_frequency=2e9, gain=30.0)
SWEEP = AIR.sensor.sweep_spec_cls(
    source=AIR.schema.source(), captures=(CAPTURE, HIGH_GAIN)
)
TONE = preset_capture('single_tone', **SCALE_ONLY)
TONE_SWEEP = preset_sweep('single_tone', (TONE,))


def delayed(extra_data, capture=CAPTURE, sweep=SWEEP, extra_coords=None):
    return make_delayed(
        extra_data, sweep=sweep, capture=capture, extra_coords=extra_coords
    )


# %% overloads


def test_overloads_one_entry_per_port_in_split_order():
    ports = summary.overloads(delayed({'adc_headroom': (1.0, 1.0)}))
    assert [p.capture.port for p in ports] == [0, 1]
    assert [p.capture.center_frequency for p in ports] == [1e9, 2e9]


@pytest.mark.parametrize(
    'headroom, adc',
    [
        ((-1.0, 3.0), [True, False]),
        ((3.0, 0.0), [False, True]),
        ((1.0, 3.0), [False, False]),
    ],
    ids=['negative_on_port_0', 'zero_on_port_1', 'positive'],
)
def test_adc_overload_is_headroom_at_or_below_zero(headroom, adc):
    ports = summary.overloads(delayed({'adc_headroom': headroom}))
    assert [p.adc for p in ports] == adc
    assert [p.im3_victims for p in ports] == [(), ()]


def test_im3_victims_follow_the_im3_rule():
    """third-order products of a channel visited at higher gain grow 2 dB per dB of
    gain difference relative to the signal. 10 dB of IF headroom at 0 dB gain is
    overrun by the 30 dB gain capture at 2 GHz on port 0 (10 + 2/3 * (0 - 30) < 0),
    but not by the 1 GHz capture at equal gain, nor on port 1."""
    ports = summary.overloads(delayed({'if_headroom': (10.0, 10.0)}))
    assert [p.im3_victims for p in ports] == [(2e9,), ()]
    assert [p.adc for p in ports] == [False, False]


def test_im3_victims_are_empty_with_ample_headroom():
    ports = summary.overloads(delayed({'if_headroom': (21.0, 21.0)}))
    assert [p.im3_victims for p in ports] == [(), ()]


@pytest.mark.parametrize(
    'extra, capture, sweep',
    [
        ({}, CAPTURE, SWEEP),
        ({'adc_headroom': (-1.0, -1.0), 'if_headroom': (-1.0, -1.0)}, TONE, TONE_SWEEP),
    ],
    ids=['no_headroom_keys', 'not_a_soapy_capture'],
)
def test_overloads_is_empty(extra, capture, sweep):
    assert summary.overloads(delayed(extra, capture, sweep)) == ()


# %% describe_overloads


def test_describe_overloads_is_none_without_an_overload():
    ports = summary.overloads(delayed({'adc_headroom': (1.0, 3.0)}))
    assert summary.describe_overloads(ports) is None
    assert summary.describe_overloads(()) is None


def test_describe_overloads_names_the_overloaded_port_and_its_victim_in_mhz():
    ports = summary.overloads(
        delayed({'adc_headroom': (-1.0, 3.0), 'if_headroom': (10.0, 10.0)})
    )
    text = summary.describe_overloads(ports)
    assert isinstance(text, str)
    assert 'port 0 (1000 MHz)' in text
    assert 'onto 2000 MHz' in text
    assert 'port 1' not in text


# %% spectrogram_snr

NOISE_BANDWIDTH = 1e4
PEAK_DBM = -100.0


def test_spectrogram_snr_is_the_peak_above_the_calibrated_noise(calibration_nc):
    """the SNR is the spectrogram peak above the calibrated system noise kTB plus the
    noise figure recorded in the calibration file for the port and center frequency
    of the capture (5 dB at port 0, 1 GHz)"""
    capture = soapy_capture()
    noise_dBm = 5.0 + 10 * math.log10(BOLTZMANN_MW * T_REF * NOISE_BANDWIDTH)

    entries = summary.spectrogram_snr(
        air_sweep(capture, calibration=calibration_nc),
        capture,
        spectrogram_result(PEAK_DBM),
    )

    assert entries is not None and len(entries) == 1
    assert entries[0].capture.port == 0
    assert entries[0].snr_dB == pytest.approx(PEAK_DBM - noise_dBm, abs=0.5)


def test_spectrogram_snr_is_none_without_a_calibration():
    capture = soapy_capture()
    result = spectrogram_result(PEAK_DBM)
    assert summary.spectrogram_snr(air_sweep(capture), capture, result) is None


def test_spectrogram_snr_skips_a_port_with_no_power(calibration_nc):
    capture = soapy_capture()
    spg, attrs = spectrogram_result(PEAK_DBM)
    spg[:] = 0.0
    entries = summary.spectrogram_snr(
        air_sweep(capture, calibration=calibration_nc), capture, (spg, attrs)
    )
    assert entries == ()


# %% capture_fields


def qualname(cls) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def field_by_name(groups) -> dict[str, summary.FieldValue]:
    return {f.name: f for g in groups for f in g.fields}


def test_capture_fields_groups_follow_the_declaring_classes():
    """`SoapyCapture` declares `center_frequency` and `gain` on top of `SensorCapture`,
    which extends `Capture`. A group is one of these classes, base first."""
    groups = summary.capture_fields(delayed({}))

    hierarchy = (sa.specs.Capture, ss.specs.SensorCapture, ss.specs.SoapyCapture)
    assert [g.owner for g in groups] == [qualname(cls) for cls in hierarchy]
    assert [f.name for f in groups[0].fields] == list(
        sa.specs.Capture.__struct_fields__
    )
    assert [f.name for f in groups[2].fields] == ['center_frequency', 'gain']


def test_capture_fields_omit_the_fields_that_are_not_coordinates():
    names = set(field_by_name(summary.capture_fields(delayed({}))))
    assert 'adjust_analysis' in ss.specs.SoapyCapture.__struct_fields__
    assert names.isdisjoint({'adjust_analysis', 'signal_trigger', 'capture_index'})


@pytest.mark.parametrize(
    'capture, port_count', [(CAPTURE, 2), (soapy_capture(port=1), 1)], ids=['2', '1']
)
def test_capture_fields_list_one_value_per_port(capture, port_count):
    fields = field_by_name(summary.capture_fields(delayed({}, capture)))
    assert {len(f.values) for f in fields.values()} == {port_count}


def test_capture_fields_list_the_port_values_in_split_order():
    fields = field_by_name(summary.capture_fields(delayed({})))
    ports = ss.specs.captures.split_capture_ports(CAPTURE)
    assert fields['port'].values == tuple(str(c.port) for c in ports)
    assert fields['port'].units == ''


@pytest.mark.parametrize(
    'name, units, values',
    [('center_frequency', 'GHz', ('1', '2')), ('sample_rate', 'MS/s', ('125', '125'))],
    ids=['center_frequency', 'sample_rate'],
)
def test_capture_fields_share_one_si_prefix_per_field(name, units, values):
    """1 GHz and 2 GHz share the giga prefix. The 125 MS/s clock of the fake device
    has the mega prefix. `Meta` of the field gives the base units."""
    fields = field_by_name(summary.capture_fields(delayed({})))
    assert fields[name] == summary.FieldValue(name, values, units)


def test_capture_fields_render_text_and_infinite_values_without_a_prefix():
    """`lo_shift` is a string with no units, and an infinite `analysis_bandwidth`
    keeps its base units"""
    capture = soapy_capture(port=(0, 1), analysis_bandwidth=math.inf)
    fields = field_by_name(summary.capture_fields(delayed({}, capture)))
    assert fields['lo_shift'] == summary.FieldValue('lo_shift', ('none', 'none'), '')
    assert fields['analysis_bandwidth'].values == ('inf', 'inf')
    assert fields['analysis_bandwidth'].units == 'Hz'


# %% acquisition_fields

INFO = ss.specs.SoapyAcquisitionInfo(
    source_id='beef',
    sweep_index=2,
    capture_index=3,
    start_time=None,
    backend_sample_rate=15.36e6,
    signal_trigger=FakeTrigger(0.0),
)


def test_acquisition_fields_groups_follow_the_declaring_classes():
    """`source_id` is redeclared by `SoapyAcquisitionInfo` and stays with
    `AcquisitionInfo`, which first declares it"""
    groups = summary.acquisition_fields(delayed({}, extra_coords=INFO))
    hierarchy = (ss.specs.AcquisitionInfo, ss.specs.SoapyAcquisitionInfo)
    assert [g.owner for g in groups] == [qualname(cls) for cls in hierarchy]
    assert [f.name for f in groups[0].fields] == [
        'source_id',
        'sweep_index',
        'capture_index',
        'burst_index',
    ]
    assert [f.name for f in groups[1].fields] == ['backend_sample_rate']


def test_acquisition_fields_skip_the_trigger_and_none_values():
    names = set(
        field_by_name(summary.acquisition_fields(delayed({}, extra_coords=INFO)))
    )
    assert INFO.signal_trigger is not None
    assert names.isdisjoint({'signal_trigger', 'start_time', 'sweep_start_time'})


def test_acquisition_fields_have_one_value_each():
    fields = field_by_name(summary.acquisition_fields(delayed({}, extra_coords=INFO)))
    assert fields['capture_index'] == summary.FieldValue('capture_index', ('3',), '')
    assert fields['sweep_index'] == summary.FieldValue('sweep_index', ('2',), '')
    assert fields['source_id'] == summary.FieldValue('source_id', ('beef',), '')
    assert fields['backend_sample_rate'] == summary.FieldValue(
        'backend_sample_rate', ('15.36',), 'MS/s'
    )


# %% analysis_labels


def test_analysis_labels_are_the_registered_long_names():
    """the registry is keyed by spec type. The expected label of each enabled field
    is read from the registration of the type of the field value."""
    expected = {
        name: sa.registry[type(getattr(ANALYSIS, name))].attrs['long_name']
        for name in ANALYSIS.__struct_fields__
        if getattr(ANALYSIS, name) is not None
    }
    assert len(expected) == 3
    assert summary.analysis_labels(ANALYSIS) == expected


def test_analysis_labels_fall_back_to_the_title_cased_field_name():
    labels = summary.analysis_labels(ANALYSIS, sa.register.AnalysisRegistry())
    assert labels == {
        'iq_waveform': 'Iq Waveform',
        'power_spectral_density': 'Power Spectral Density',
        'channel_power_time_series': 'Channel Power Time Series',
    }
