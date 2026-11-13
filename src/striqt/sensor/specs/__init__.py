"""schema for the specification of calibration and sweeps"""

# the loop below rewrites the __module__ of each struct to this package, and msgspec
# then resolves the string annotations of the structs in this namespace. The modules
# that the annotations name must therefore stay bound here. 'pd' also stands in for
# the python 3.15 lazy import of pandas

from __future__ import annotations as __

from .structs import *

# only classes, not modules
for k in list(locals().keys()):
    if k in ('pd', 'msgspec'):
        continue
    if k[0].upper() != k[0]:
        locals().pop(k)
del k

from striqt.analysis.specs import frozendict

from . import action, captures, paths, schema, sequencing, structs, types
from .dataclasses import AcquiredIQ, PendingExtraData, Schema
from .structs import SA, SC, SP, SS
from .types import Annotated, Meta

for k, obj in list(locals().items()):
    if k in ('pd', 'msgspec'):
        continue
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__

del k, obj
