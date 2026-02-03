"""schema for the specification of calibration and sweeps"""

from __future__ import annotations as __

from . import helpers, types
from .structs import *
from .structs import _TS, _TP, _TC, _TPC
from .types import Annotated, Meta
from striqt.analysis.specs import immutabledict