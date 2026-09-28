"""what the type checker sees at striqt's keyword-unpacking sites.

Measurement signatures are synthesized (`hint_keywords` binds a `ParamSpec` from the
spec Struct), coordinate factories, `Controller` construction and `arm`, the cache and
`retry` decorators re-scope their type variables through generic wrappers, and
`design_resampler`/`iterate_sweep` forward `**kwargs: Unpack[TypedDict]`. None of that
runs at test time, so a broken Protocol or a lost `ParamSpec` would be silent.

Each probe under `tests/ty_probes/` is ordinary typed code that must check clean: a
call that must be accepted asserts its result with `typing_extensions.assert_type`, and
a call that must be rejected carries `# ty: ignore[<rule>]`, which ty reports as an
unused ignore if the rejection ever stops. The probe for every registered measurement is
generated here from `sa.registry` on the same pattern.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import msgspec
import pytest

import striqt.analysis as sa

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_DIR = pathlib.Path(__file__).with_name('ty_probes')


def ty_check(target: pathlib.Path | str) -> subprocess.CompletedProcess[str]:
    # --project is required: without it ty takes the target's directory as the
    # project, so [tool.ty] in pyproject.toml does not apply
    command = [
        sys.executable,
        '-m',
        'ty',
        'check',
        str(target),
        '--output-format',
        'concise',
        '--project',
        str(REPO_ROOT),
        '--python',
        sys.prefix,
    ]
    return subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )


def assert_checks_clean(target: pathlib.Path | str) -> None:
    proc = ty_check(target)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# %% probes under tests/ty_probes
@pytest.mark.parametrize(
    'probe', sorted(PROBE_DIR.glob('probe_*.py')), ids=lambda p: p.stem[len('probe_') :]
)
def test_probe_checks_clean(probe: pathlib.Path) -> None:
    assert_checks_clean(probe)


@pytest.mark.xfail(
    strict=True,
    reason='ty 0.0.81 lowers **kwargs: Unpack[TypedDict] to **kwargs: object and '
    'accepts unknown keywords',
)
def test_unknown_keyword_through_unpack_is_flagged() -> None:
    assert_checks_clean(PROBE_DIR / 'xfail_unpack_unknown_keyword.py')


# %% hint_keywords
def measurement_call(func: str, required: list[str], *extra: str) -> str:
    """a call of `func` with every `required` field satisfied by `anything()`, unless
    the field is named again in `extra`"""
    kws = [f'{name}=anything()' for name in required if not _named(name, extra)]
    return f'{func}(iq, cap, {", ".join([*kws, *extra])})'


def _named(name: str, extra: tuple[str, ...]) -> bool:
    return any(e.startswith(f'{name}=') for e in extra)


def registry_probe() -> str:
    """a probe calling every registered measurement: valid calls must return the
    `as_xarray` types, an unknown keyword must be rejected, and each spec field must
    reject an `object()`, which proves that the field is a known, typed parameter"""
    modules = sorted({info.func.__module__ for info in sa.registry.values()})
    lines = [
        'from typing import Any',
        'from typing_extensions import assert_type',
        'import numpy as np',
        'import xarray as xr',
        'import striqt.analysis as sa',
        'from striqt.analysis.lib.dataarrays import DelayedDataArray',
        *[f'import {module}' for module in modules],
        '',
        '',
        'def anything() -> Any: ...',
        '',
        '',
        'def probe(iq: np.ndarray, cap: sa.specs.Capture) -> None:',
    ]
    for spec_type, info in sa.registry.items():
        # <module>.<name> rather than sa.measurements.<name>: cellular_5g_sss_sync is
        # registered but not exported
        func = f'{info.func.__module__}.{info.name}'
        fields = msgspec.structs.fields(spec_type)
        required = [f.name for f in fields if f.required]
        lines += [
            f'    assert_type({measurement_call(func, required)}, xr.DataArray)',
            '    assert_type(',
            f'        {measurement_call(func, required, "as_xarray=" + repr("delayed"))},',
            '        DelayedDataArray,',
            '    )',
            (
                f'    {measurement_call(func, required, "__bogus__=1")}'
                '  # ty: ignore[no-matching-overload]'
            ),
            *[
                (
                    f'    {measurement_call(func, required, f.name + "=object()")}'
                    '  # ty: ignore[invalid-argument-type]'
                )
                for f in fields
            ],
        ]
    return '\n'.join(lines) + '\n'


def test_every_measurement_exposes_its_spec_fields(tmp_path: pathlib.Path) -> None:
    probe = tmp_path / 'probe_registry.py'
    probe.write_text(registry_probe())
    assert_checks_clean(probe)


# %% ty check src
@pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason='ty check src is kept clean on the py314 environment only; test39 has a '
    'known baseline from zarr 2.x and older stubs',
)
def test_src_has_no_diagnostics() -> None:
    assert_checks_clean('src')
