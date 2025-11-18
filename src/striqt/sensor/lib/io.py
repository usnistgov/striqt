"""just a stub for now in case we change this in the future"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

from striqt import analysis
from striqt.analysis.lib.io import decode_from_yaml_file, dump, load  # noqa: F401
from striqt.analysis.lib.specs import convert_dict

from . import captures, specs, util

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


def open_store(
    spec: specs.Sink,
    *,
    alias_func: captures.PathAliasFormatter | None = None,
    force=False,
):
    spec_path = spec.path

    if alias_func is not None:
        spec_path = alias_func(spec_path)

    spec_path = Path(spec_path)

    if spec.store == 'directory' and not spec_path.stem.endswith('.zarr'):
        fixed_path = spec_path.with_suffix('.zarr')
    elif spec.store == 'zip' and not spec_path.stem.endswith('.zarr.zip'):
        fixed_path = spec_path.with_suffix('.zarr.zip')
    else:
        fixed_path = spec_path

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = analysis.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def read_yaml_sweep(
    path: str | Path,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str] = None,
) -> specs.Sweep:
    """build a Sweep specification object from the specified yaml file.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of structs.SweepSpec (or subclass)
    """

    from .bindings import get_tagged_sweep_spec

    # first pass is a simple dict
    tree = decode_from_yaml_file(path)

    if not isinstance(tree, dict):
        raise TypeError('yaml file does not specify a dict structure')

    spec = convert_dict(tree, type=get_tagged_sweep_spec())
    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    return spec.replace(output=sink)


def read_tdms_iq(
    path: Path | str,
    duration: float | None = None,
    *,
    base_clock_rate,
    num_rx_ports=1,
    dtype='complex64',
    skip_samples=0,
    array_backend: specs.ArrayBackendType,
) -> tuple['np.ndarray', specs.FileCapture]:
    from .sources.file import TDMSFileSource, TDMSSourceSpec

    source_spec = TDMSSourceSpec(
        base_clock_rate=base_clock_rate, path=Path(path), num_rx_ports=num_rx_ports
    )
    source = TDMSFileSource(source_spec)

    capture = source.capture_spec

    source.arm(capture)
    iq, _ = source.read_iq()

    return iq, capture
