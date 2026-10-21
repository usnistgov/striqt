"""schema for the specification of calibration and sweeps"""

# the structs' __module__ is rewritten to this package below, and msgspec resolves
# their string annotations in this namespace, so the modules those annotations name
# must stay bound here. 'pd' also stands in for python 3.15 lazy imports

from __future__ import annotations as __

from .structs import *

# only classes, not modules
for k in list(locals().keys()):
    if k in ('pd', 'msgspec'):
        continue
    if not k[0].upper() == k[0]:
        locals().pop(k)
del k

from . import captures, paths, sequencing, structs, sweep, types
from .dataclasses import AcquiredIQ, Schema
from .structs import SS, SP, SC
from .types import Annotated, Meta
from striqt.analysis.specs import frozendict

for k, obj in list(locals().items()):
    if k in ('pd', 'msgspec'):
        continue
    if getattr(obj, '__module__', '').startswith(__name__):
        obj.__module__ = __name__

del k, obj
