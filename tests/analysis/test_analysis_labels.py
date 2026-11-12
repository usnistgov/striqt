"""striqt.analysis.lib.labels: the `Measurand` built from attrs or capture field
metadata, and the quantity, units, and value text composed from it."""

from __future__ import annotations

import math

import numpy as np
import pytest

import striqt.analysis as sa

labels = sa.labels

PSD_ATTRS = {
    'standard_name': 'PSD',
    'long_name': 'Power spectral density',
    'units': 'dBm',
    'noise_bandwidth': 360e3,
}


# %% labels.measurand_from_attrs


def test_measurand_from_attrs_reads_the_4_label_keys():
    m = labels.measurand_from_attrs({**PSD_ATTRS, 'sample_rate': 1e6}, 'spectrogram')

    assert m == labels.Measurand(
        name='spectrogram',
        standard_name='PSD',
        long_name='Power spectral density',
        units='dBm',
        noise_bandwidth=360e3,
    )


@pytest.mark.parametrize(
    'bandwidth',
    [360000, np.float32(360e3), np.int64(360000)],
    ids=['int', 'f32', 'i64'],
)
def test_measurand_from_attrs_coerces_noise_bandwidth_to_float(bandwidth):
    m = labels.measurand_from_attrs({'units': 'dBm', 'noise_bandwidth': bandwidth})

    assert type(m.noise_bandwidth) is float
    assert m.noise_bandwidth == pytest.approx(360e3)


def test_measurand_from_attrs_without_keys():
    """a missing key is None, and a missing name is the empty string"""
    m = labels.measurand_from_attrs({})

    assert m == labels.Measurand(name='')
    assert m.units is None
    assert m.noise_bandwidth is None


def test_measurand_from_field_reads_the_capture_meta():
    m = labels.measurand_from_field(sa.specs.Capture, 'sample_rate')

    assert m.name == 'sample_rate'
    assert m.units == 'S/s'
    assert m.noise_bandwidth is None

    with pytest.raises(KeyError):
        labels.measurand_from_field(sa.specs.Capture, 'not_a_field')


# %% labels.units_label


@pytest.mark.parametrize(
    'attrs,prefix,expected',
    [
        ({'units': 'dBm', 'noise_bandwidth': 360e3}, '', 'dBm/360 kHz'),
        ({'units': 'dBm', 'noise_bandwidth': 10e6}, '', 'dBm/10 MHz'),
        ({'units': '√mW', 'noise_bandwidth': 30e3}, '', '√mW/30 kHz'),
        ({'units': 'dBm/Hz', 'noise_bandwidth': 360e3}, '', 'dBm/Hz'),
        ({'units': 'dBm'}, '', 'dBm'),
        ({'units': 'Hz'}, 'M', 'MHz'),
        ({'units': 'S/s'}, 'M', 'MS/s'),
        ({'units': 'dBm', 'noise_bandwidth': 360e3}, 'k', 'dBm/360 kHz'),
        ({}, '', None),
    ],
    ids=[
        'kHz_bandwidth',
        'MHz_bandwidth',
        'wave_units',
        'density_keeps_units',
        'no_bandwidth',
        'prefix_on_Hz',
        'prefix_on_rate',
        'prefix_dropped_on_dB',
        'no_units',
    ],
)
def test_units_label(attrs, prefix, expected):
    assert (
        labels.units_label(labels.measurand_from_attrs(attrs), prefix=prefix)
        == expected
    )


# %% labels.quantity_label


@pytest.mark.parametrize(
    'kws,expected',
    [
        ({}, 'Power spectral density (dBm/360 kHz)'),
        ({'short': True}, 'PSD (dBm/360 kHz)'),
        ({'with_units': False}, 'Power spectral density'),
        ({'short': True, 'with_units': False}, 'PSD'),
    ],
    ids=['long', 'short', 'long_no_units', 'short_no_units'],
)
def test_quantity_label_selects_the_name_and_appends_units_once(kws, expected):
    text = labels.quantity_label(
        labels.measurand_from_attrs(PSD_ATTRS, 'spectrogram'), **kws
    )

    assert text == expected
    assert text.count('(') == expected.count('(')


def test_quantity_label_prefix_scales_the_units():
    m = labels.measurand_from_attrs({'standard_name': 'Frequency', 'units': 'Hz'}, 'f')

    assert labels.quantity_label(m, prefix='M') == 'Frequency (MHz)'
    assert labels.quantity_label(m, prefix='G', with_units=False) == 'Frequency'


def test_quantity_label_falls_back_from_long_name_to_standard_name_to_name():
    assert labels.quantity_label(labels.Measurand('x', 'short', 'long')) == 'long'
    assert (
        labels.quantity_label(labels.Measurand('x', 'short', 'long'), short=True)
        == 'short'
    )
    assert labels.quantity_label(labels.Measurand('x', 'short', None)) == 'short'
    assert (
        labels.quantity_label(labels.Measurand('x', None, 'long'), short=True) == 'long'
    )
    assert labels.quantity_label(labels.Measurand('x')) == 'x'
    assert labels.quantity_label(labels.Measurand('x', units='Hz')) == 'x (Hz)'


# %% labels.describe_value


@pytest.mark.parametrize(
    'value,units,expected',
    [
        (None, 'Hz', 'None'),
        (3.75e9, 'Hz', '3.75 GHz'),
        ((3.7e9, 3.75e9), 'Hz', '(3700 MHz, 3750 MHz)'),
        ((1e6, None), 'S/s', '(1 MS/s, None)'),
        (-20.0, 'dBm', '-20 dBm'),
        ('rx1', 'Hz', "'rx1'"),
        (math.inf, 'Hz', 'inf'),
        (math.nan, 'Hz', 'nan'),
        (True, 'Hz', 'True'),
        (2.5e6, None, '2500000.0'),
    ],
    ids=[
        'none',
        'single',
        'tuple_shared_prefix',
        'tuple_with_none',
        'dB',
        'str',
        'inf',
        'nan',
        'bool',
        'no_units',
    ],
)
def test_describe_value(value, units, expected):
    assert labels.describe_value(value, units) == expected
    assert labels.describe_value(value, labels.Measurand('x', units=units)) == expected


def test_describe_value_forced_prefix():
    assert labels.describe_value(3.75e9, 'Hz', prefix='M') == '3750 MHz'
    assert (
        labels.describe_value((3.7e9, 3.75e9), 'Hz', prefix='G')
        == '(3.7 GHz, 3.75 GHz)'
    )


# %% labels.describe_field


def test_describe_field_names_the_value_in_its_units():
    capture = sa.specs.Capture(sample_rate=1e6, duration=1e-3)

    assert labels.describe_field(capture, 'sample_rate') == 'sample_rate=1 MS/s'
    assert labels.describe_field(capture, 'duration', sep=': ') == 'duration: 1 ms'

    with pytest.raises(KeyError):
        labels.describe_field(capture, 'not_a_field')
