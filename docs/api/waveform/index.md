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
        - is_cupy_array
        - configure_cupy
        - pinned_array_as_cupy
        - axis_index
        - axis_slice
        - axis_to_blocks
        - binned_mean
        - histogram_last_axis
        - isroundmod
        - cp

## Fourier analysis and resampling

FFT and STFT evaluation, window design, overlap-add convolution, and rational resampling.

::: striqt.waveform
    options:
      extra:
        module_docstring: false
      members:
        - fft
        - ifft
        - fftfreq
        - stft
        - istft
        - spectrogram
        - get_window
        - equivalent_noise_bandwidth
        - oaconvolve
        - oaresample
        - resample
        - design_cola_resampler
        - design_fir_lpf
        - ResamplerDesign
        - get_max_cupy_fft_chunk
        - set_max_cupy_fft_chunk

## Power

Power detectors, empirical distributions, and conversions between dB, power, and envelope.

::: striqt.waveform
    options:
      extra:
        module_docstring: false
      members:
        - iq_to_bin_power
        - iq_to_cyclic_power
        - sample_ccdf
        - dBtopow
        - powtodB
        - envtodB
        - envtopow
        - dBlinmean
        - dBlinsum
