"""striqt.analysis.specs.types: the semantic field type aliases"""

from __future__ import annotations

from typing import Optional, Union, get_args

import pytest
from conftest import construct_both
from msgspec import Meta

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis.specs.doc import parse_type_and_meta
from striqt.analysis.specs.types import PER_PORT, PerPort, is_per_port

T = ss.specs.types

# %% PerPort


def test_per_port_is_the_union_of_the_scalar_and_its_tuple_with_the_marker():
    base, *metadata = get_args(PerPort[int])
    assert base == Union[int, tuple[int, ...]]
    assert metadata == [PER_PORT]


@pytest.mark.parametrize(
    'hint, expected',
    [
        (PerPort[float], True),
        (T.Gain, True),
        (T.Port, True),
        (T.LOFrequency, True),
        (T.CenterFrequency, True),
        (Optional[PerPort[str]], True),
        (T.GainScalar, False),
        (T.CellularSubcarrierSpacingTuple, False),
        (Union[float, tuple[float, ...]], False),
        (float, False),
    ],
    ids=str,
)
def test_is_per_port_reads_the_marker_through_annotated_and_optional(hint, expected):
    assert is_per_port(hint) is expected


def test_parse_type_and_meta_keeps_the_meta_and_drops_the_marker():
    clean, metas = parse_type_and_meta(T.Gain)
    assert clean == Union[float, tuple[float, ...]]
    assert all(isinstance(m, Meta) for m in metas)
    assert metas[0].description == 'Gain setting for each channel'


def test_capture_type_attrs_of_a_per_port_field():
    attrs = sa.specs.helpers.get_capture_type_attrs(ss.specs.SoapyCapture)
    assert attrs['gain'] == {
        'standard_name': 'Gain setting for each channel',
        'units': 'dB',
    }


@pytest.mark.parametrize(
    'kws',
    [
        {'port': 0, 'center_frequency': 1e9, 'gain': -10.0},
        {'port': (0, 1), 'center_frequency': (1e9, 2e9), 'gain': (0.0, 5.0)},
        {'port': (0, 1), 'center_frequency': 1e9, 'gain': (0.0, 5.0)},
    ],
    ids=['scalars', 'tuples', 'mixed'],
)
def test_per_port_fields_decode_scalars_and_tuples_on_both_paths(kws):
    direct, converted = construct_both(ss.specs.SoapyCapture, **kws)
    assert direct == converted
    assert (direct.port, direct.center_frequency, direct.gain) == tuple(kws.values())
