"""schema for the specification of calibration and sweeps"""

# note: 'pd' special cases can be removed in the future
# with python 3.15 lazy imports

from __future__ import annotations as __

from .structs import *

# only classes, not modules
for k in list(locals().keys()):
    if k == 'pd':
        continue
    if not k[0].upper() == k[0]:
        locals().pop(k)
del k

from . import helpers, types
from .dataclasses import AcquiredIQ, Schema
from .structs import SS, SP, SC, SPC
from .types import Annotated, Meta
from striqt.analysis.specs import frozendict

for k, obj in list(locals().items()):
    if k == 'pd':
        continue
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__

del k, obj
