# Waveform numerics

`striqt.waveform` is the numerical layer under the measurement packages. Every
function in the package accepts numpy and cupy arrays and computes in 32-bit
float. Paths below are relative to `src/striqt/waveform/` unless qualified.

## Backend dispatch

- `sw.array_namespace(x)` returns the array module of `x`. A function that must
  branch on backend uses that module rather than a hard-coded `numpy`.
- `sw.is_cupy_array(x)` takes an array, not a namespace. `is_cupy_array(xp)` on a
  module is always False and silently disables the GPU path.
- The elementwise dB conversions in `lib/power/convert.py` evaluate with numexpr on
  numpy and with the `cupy.fuse` kernels in `lib/power/cuda_jit.py` on cupy. The
  two paths must stay in lockstep, including the `out=` handling through
  `_arraylike_with_buffer`.
- `comms.correlate_cyclic_shift` dispatches the same way, between
  `lib/comms/cp_cpu_jit.py` (numba `njit`) and `lib/comms/cp_cuda_jit.py`
  (`numba.cuda.jit`).
- `lib/jit.py` sets `NUMBA_CACHE_DIR` to the platformdirs cache directory when the
  variable is unset, and stamps numba caches by content hash so that a reinstall
  does not invalidate the caches. The two kernel modules import `lib/jit.py` before
  numba so that the setting takes effect.
- coverage.py excludes the jitted and fused bodies through `exclude_also` in the
  `[tool.coverage.report]` table of `pyproject.toml`.

## Fourier package

`lib/fourier/` is split by concern. The modules below are listed in dependency order.

- `windows.py`: `get_window` and the disk cache of window designs, the ENBW search
  (`find_window_param_from_enbw`), and the extra scipy-style windows.
- `fft.py`: `fft` and `ifft` with the cupy `out=` path, the exact `fftfreq` grid,
  `slice_freqs`, `truncate_freqs`, `null_lo`, `time_fftshift`, and the roundoff
  model `fft_tolerance_rms`, `on_peak_roundoff`, and `off_peak_floor_dBc`.
- `stft.py`: `stft` and `istft`, the stacking helpers, `zero_stft_by_freq`, and
  `downsample_stft`.
- `spectrum.py`: `spectrogram`.
- `filters.py`: `design_oafilter` and the COLA divisor table, `design_fir_lpf`,
  `oafilter`, and `oaconvolve`.
- `resamplers.py`: `design_cola_resampler`, `resample`, and `oaresample`.

`lib/fourier/__init__.py` imports only the submodules, so inside `lib` the names are
reached as `fourier.stft.stft`, `fourier.windows.get_window`, and so on. The public
shim is `striqt.waveform.fourier`.

## STFT layout

- `stft(x, axis=a)` returns windows stacked on axis `a` and frequency bins on axis
  `a + 1`.
- The bins are in ascending `fftfreq` order on return. The shift is baked into the
  window through `get_window(..., fftshift=True)`, so no `fftshift` follows the FFT.
- `noverlap` must be 0 or `nperseg // 2`. Other values raise.
- `fftfreq` computes the bin frequencies with `decimal`, so the grid is exact.
- Band selections through `_freq_band_edges` (`zero_stft_by_freq`,
  `downsample_stft`) are half-open, `[lo, hi)`, on that grid.
- The `out=` arguments of `istft` and `downsample_stft` are honored through
  `_truncated_buffer`. `_truncated_buffer` returns a view of a contiguous buffer and
  copies only a non-contiguous one.

## Resampling

- `design_cola_resampler` returns a `ResamplerDesign`. The ratio `nfft / nfft_out`
  equals `fs_sdr / fs_target` exactly, and both sizes are divisible by the COLA
  divisor of the window.
- The divisor table is `_COLA_WINDOW_SIZE_DIVISOR` in `filters.py`. The sensor
  resamples with the hamming window, `RESAMPLE_COLA_WINDOW` in
  `striqt/sensor/lib/compute/corrections.py`, whose divisor is 2.
- `oaresample` and `resample` implement the design through STFT overlap-add.
  `oafilter` is the same machinery used as a low-pass.
- `resample` fftshifts in the time domain (`time_fftshift`) so that downsampling
  becomes a slice of the frequency axis.

## Caching

- `lib/util.py` provides `lru_cache`, a wrapper over `functools.lru_cache` that
  records each wrapped function in `_caches`. `clear_caches()` clears every
  recorded cache.
- `lib/util.py` also provides `persistent_cache`, a `shelve` file under the
  platformdirs cache directory. The window design searches (`get_window`,
  `find_window_param_from_enbw`) use the shelf so that a design survives across
  processes.
- `persistent_cache` sits under `lru_cache` in `windows.py` but is not itself LRU.
  The access order is rebuilt from the shelf keys on every call, so the entry
  evicted past `maxsize` is whichever key the dbm backend iterates first.
- The test suite replaces the shelf with an in-memory dict through the
  session-scoped `isolated_persistent_cache` fixture in `tests/conftest.py`. The
  macOS dbm backend corrupts under eviction and under concurrent writes from 2
  processes, and a corrupted shelf fails every later `get_window` call on the
  machine.

## OFDM and cellular

`lib/comms/` holds the OFDM frame models and the 5G NR synchronization.

- `core.py`: `PhyOFDM`, the frame model shared by the standards below.
- `phy_3gpp.py`: `Phy3GPP`, for LTE and 5G NR. The cyclic prefix sizes and start
  indices follow 3GPP TS 38.211 §5.3.1.
- `phy_802_16.py`: `Phy802_16`.
- `cp.py`: `correlate_cyclic_shift`.
- `sequences.py`: `m_sequence`.
- `sync_5gnr.py`: `index_pss_symbols` encodes the SSB cases of 3GPP TS 38.213
  §4.1. `correlate_sync_sequence` returns a grid over (coarse lag, fine lag), with
  the axes named by `COARSE_LAG_DIM` and `FINE_LAG_DIM`.

The tests validate the synchronization path at 30 kHz subcarrier spacing only. The
cyclic prefix model at 15 kHz and 60 kHz is known wrong. The defect is recorded as
strict xfails in `tests/waveform/test_waveform_comms_phy_3gpp.py`, and a fix changes
the shape of `Phy3GPP.cp_sizes`.
