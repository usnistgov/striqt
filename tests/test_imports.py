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


def test_lazy_import():
    # a fresh interpreter isolates the check from conftest imports, pytest
    # plugins, and (on python 3.9) linecache scanning sys.modules for __file__
    result = subprocess.run(
        [sys.executable, '-c', LAZY_IMPORT_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_striqt_figures():
    import striqt.figures  # noqa: F401
