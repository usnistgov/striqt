# striqt.waveform

::: striqt.waveform
    options:
      members: false

## Arrays and backends

Array-namespace dispatch between numpy and cupy, axis helpers, and the cupy configuration.

::: striqt.waveform
    options:
      extra:
        module_docstring: false
      members:
        - array_namespace
        - axis_index
        - axis_slice
        - axis_to_blocks
        - binned_mean
        - configure_cupy
        - histogram_last_axis
        - is_cupy_array
        - isroundmod
        - pinned_array_as_cupy

## Fourier analysis and resampling

FFT and STFT evaluation, window design, overlap-add convolution, and rational resampling.

::: striqt.waveform
    options:
      extra:
        module_docstring: false
      members:
        - ResamplerDesign
        - design_cola_resampler
        - design_fir_lpf
        - equivalent_noise_bandwidth
        - fft
        - fftfreq
        - get_max_cupy_fft_chunk
        - get_window
        - ifft
        - istft
        - oaconvolve
        - oaresample
        - resample
        - set_max_cupy_fft_chunk
        - spectrogram
        - stft

## Power

Power detectors, empirical distributions, and conversions between dB, power, and envelope.

::: striqt.waveform
    options:
      extra:
        module_docstring: false
      members:
        - dBlinmean
        - dBlinsum
        - dBtopow
        - envtodB
        - envtopow
        - iq_to_bin_power
        - iq_to_cyclic_power
        - powtodB
        - sample_ccdf
