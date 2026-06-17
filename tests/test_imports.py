"""these tests run first, before other tests reify lazy imports"""

import pytest


# @pytest.mark.order(0)
# def test_striqt_waveform():
#     import striqt.waveform


# @pytest.mark.order(1)
# def test_striqt_analysis():
#     import striqt.analysis


# @pytest.mark.order(2)
# def test_striqt_sensor():
#     import striqt.sensor


# @pytest.mark.order(3)
# def test_striqt_cli():
#     import striqt.cli


@pytest.mark.order(0)
def test_lazy_import():
    import striqt.cli
    import striqt.waveform
    import striqt.analysis
    import striqt.sensor
    import importlib.util

    # check for accidental reification
    import sys

    # Note: scipy, xarray, and pandas may be reified by the hypothesis pytest
    # plugin during test collection when it calls _get_local_constants() which
    # accesses __file__ on all modules in sys.modules. This is a known issue
    # with hypothesis and lazy imports. We skip these checks when hypothesis
    # tests are collected (detected by checking if hypothesis has been imported).
    hypothesis_active = 'hypothesis' in sys.modules
    
    # lazy module checks - skip if hypothesis plugin is active (it reifies them)
    if not hypothesis_active:
        for lazy_name in ('scipy', 'xarray', 'pandas'):
            # don't test zarr here, since some pytest plugins already reify it
            assert 'LazyModule' in repr(type(sys.modules[lazy_name])), (
                f'reified {lazy_name}'
            )

    for name in ('dask', 'dask.array'):
        assert name not in sys.modules, f'accidental import of {name}'


def test_striqt_figures():
    """this reifies scipy; put it after lazy_import"""
    import striqt.figures
