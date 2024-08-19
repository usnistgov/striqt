"""just a stub for now in case we change this in the future"""

from __future__ import annotations
from channel_analysis import load, dump
import msgspec
from .structs import Sweep
from pathlib import Path

__all__ = ['load', 'dump', 'read_yaml_sweep']


def read_yaml_sweep(path: str | Path) -> tuple[Sweep, tuple[str, ...]]:
    """build a Sweep struct from the contents of specified yaml file"""

    with open(path, 'rb') as fd:
        text = fd.read()

    # validate first
    msgspec.yaml.decode(text, type=Sweep, strict=False)

    # build a dict to extract the list of sweep fields and apply defaults
    tree = msgspec.yaml.decode(text, type=dict, strict=False)
    sweep_fields = sorted(set.union(*[set(c) for c in tree['captures']]))

    # apply default capture settings
    defaults = tree['defaults']
    if tree['radio_setup'].get('calibration', None):
        cal_path = Path(tree['radio_setup']['calibration'])
        sweep_parent = Path(path).parent
        if cal_path.is_relative_to(sweep_parent):
            # take relative paths with respect to a yaml file,
            # not the interpreter working directory
            cal_path = cal_path.relative_to(sweep_parent)
        cal_path = str(cal_path)
        tree['radio_setup']['calibration'] = cal_path

        # # read to validate the data and warm the calibration cache
        # iq_corrections.read_calibration_corrections(cal_path)

    tree['captures'] = [dict(defaults, **c) for c in tree['captures']]

    run = msgspec.convert(tree, type=Sweep, strict=False)

    return run, sweep_fields
