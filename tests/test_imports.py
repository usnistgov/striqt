"""import-time side effects of the striqt packages"""

import json
import subprocess
import sys
import textwrap

LAZY_IMPORT_CHECK = textwrap.dedent(
    """
    import json
    import sys
    import striqt.cli
    import striqt.waveform
    import striqt.analysis
    import striqt.sensor

    # zarr is not checked: some pytest plugins reify it
    names = ('scipy', 'xarray', 'pandas', 'dask', 'dask.array')
    print(json.dumps({n: repr(type(sys.modules.get(n))) if n in sys.modules else None for n in names}))
    """
)

FIGURES_IMPORT_CHECK = textwrap.dedent(
    """
    import json
    import sys
    import striqt.figures

    names = (
        'scipy', 'xarray', 'pandas', 'matplotlib', 'dask.array',
        'striqt.figures.ticker', 'striqt.figures.notebook_env',
    )
    print(json.dumps({n: repr(type(sys.modules.get(n))) if n in sys.modules else None for n in names}))
    """
)


def module_types_in_fresh_interpreter(code: str) -> dict:
    # a fresh interpreter isolates the check from conftest imports, pytest
    # plugins, and (on python 3.9) linecache scanning sys.modules for __file__
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def assert_lazy(subtests, types: dict, names):
    for name in names:
        with subtests.test(msg=f'{name} stays lazy'):
            assert types[name] is not None and 'LazyModule' in types[name], (
                f'reified {name}: {types[name]}'
            )


def assert_absent(subtests, types: dict, names):
    for name in names:
        with subtests.test(msg=f'{name} is not imported'):
            assert types[name] is None, f'accidental import of {name}'


def test_lazy_import(subtests):
    types = module_types_in_fresh_interpreter(LAZY_IMPORT_CHECK)
    assert_lazy(subtests, types, ('scipy', 'xarray', 'pandas'))
    assert_absent(subtests, types, ('dask', 'dask.array'))


def test_striqt_figures(subtests):
    types = module_types_in_fresh_interpreter(FIGURES_IMPORT_CHECK)
    # the parent 'dask' package is reified because find_spec('dask.array')
    # imports it; only the submodule stays lazy
    assert_lazy(
        subtests, types, ('scipy', 'xarray', 'pandas', 'matplotlib', 'dask.array')
    )
    # these import matplotlib and pandas eagerly, so the check above holds only
    # while the package does not import them
    assert_absent(
        subtests, types, ('striqt.figures.ticker', 'striqt.figures.notebook_env')
    )
