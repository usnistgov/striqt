def test_striqt_waveform():
    import striqt.waveform

def test_striqt_analysis():
    import striqt.analysis

def test_striqt_sensor():
    import striqt.sensor

def test_striqt_cli():
    import striqt.cli

def test_lazy_import():
    import striqt.waveform
    import striqt.analysis
    import striqt.sensor
    import importlib.util

    # check for accidental reification
    import sys
    for lazy_name in ('scipy', 'xarray', 'pandas'):
        # don't test zarr here, since some pytest plugins already reify it
        assert 'LazyModule' in repr(type(sys.modules[lazy_name])), f"reified {lazy_name}"

def test_striqt_figures():
    """this reifies scipy; put it after lazy_import"""
    import striqt.figures
