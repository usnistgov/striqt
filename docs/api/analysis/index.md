# striqt.analysis

::: striqt.analysis
    options:
      members: false

## Evaluation and storage

Evaluation of measurement specs against IQ captures into xarray objects, the measurement registry, and zarr stores.

::: striqt.analysis
    options:
      extra:
        module_docstring: false
      members:
        - analyze_by_spec
        - AcquiredIQ
        - EvaluationOptions
        - Preprocessor
        - AnalysisRegistry
        - registry
        - Trigger
        - Capture
        - dump
        - load
        - open_store

## Measurements

The registered measurements, each evaluated from an IQ capture into one `xarray.DataArray`.

::: striqt.analysis
    options:
      extra:
        module_docstring: false
      members:
        - power_spectral_density
        - spectrogram
        - spectrogram_histogram
        - spectrogram_ratio_histogram
        - channel_power_histogram
        - channel_power_time_series
        - cyclic_channel_power
        - iq_waveform
        - cellular_5g_pss_correlation
        - cellular_5g_pss_sync
        - cellular_5g_sss_correlation
        - cellular_5g_ssb_spectrogram
        - cellular_cyclic_autocorrelation
        - cellular_resource_power_histogram
