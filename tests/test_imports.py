"""import-time side effects of the striqt packages"""

import subprocess
import sys
import textwrap

LAZY_IMPORT_CHECK = textwrap.dedent(
    """
    import sys
    import striqt.cli
    import striqt.waveform
    import striqt.analysis
    import striqt.sensor

    # zarr is not checked: some pytest plugins reify it
    for name in ('scipy', 'xarray', 'pandas'):
        assert 'LazyModule' in repr(type(sys.modules[name])), f'reified {name}'

    for name in ('dask', 'dask.array'):
        assert name not in sys.modules, f'accidental import of {name}'
    """
)

FIGURES_IMPORT_CHECK = textwrap.dedent(
    """
    import sys
    import striqt.figures

    # the parent 'dask' package is reified because find_spec('dask.array')
    # imports it; only the submodule stays lazy
    for name in ('scipy', 'xarray', 'pandas', 'matplotlib', 'dask.array'):
        assert 'LazyModule' in repr(type(sys.modules[name])), f'reified {name}'

    # these import matplotlib and pandas eagerly, so the check above holds only
    # while the package does not import them
    for name in ('striqt.figures.ticker', 'striqt.figures.notebook_env'):
        assert name not in sys.modules, f'eager import of {name}'
    """
)


def run_in_fresh_interpreter(code: str):
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


def test_lazy_import():
    run_in_fresh_interpreter(LAZY_IMPORT_CHECK)


def test_striqt_figures():
    run_in_fresh_interpreter(FIGURES_IMPORT_CHECK)
