def test_imports():
    import striqt.waveform
    import striqt.analysis
    import striqt.sensor
    import importlib.util

    # check for accidental reification
    import sys
    for lazy_name in ('scipy', 'xarray', 'pandas'):
        # don't test zarr here, since some pytest plugins already reify it
        assert 'LazyModule' in repr(type(sys.modules[lazy_name])), f"reified {lazy_name}"