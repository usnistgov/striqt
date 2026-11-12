"""a PerPort field is the union of its scalar and the tuple of that scalar"""

from __future__ import annotations

from typing import Union

from typing_extensions import assert_type

import striqt.sensor as ss


def probe(capture: ss.specs.SoapyCapture, x: ss.specs.types.PerPort[int]) -> None:
    assert_type(capture.gain, Union[float, tuple[float, ...]])
    assert_type(capture.port, Union[int, tuple[int, ...]])
    assert_type(x, Union[int, tuple[int, ...]])
