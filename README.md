[![License](https://img.shields.io/badge/license-NIST-blue)](https://github.com/usnistgov/striqt/blob/main/LICENSE.md)
![Last commit](https://img.shields.io/github/last-commit/usnistgov/striqt)
[![Documentation](https://img.shields.io/badge/Docs-purple)](https://pages.nist.gov/striqt/)

`striqt` is a set of open-source python packages for batched real-time signal analysis on CPU or GPU, and rapid prototyping of bespoke radio environment sensors.

### Basic CLI usage
🏃 Run a batched acquisition and analysis for measurement or calibration, given a path to a YAML input specification:

```sh
sensor-sweep [OPTIONS] YAML_PATH

#  Selected options:
#  -d, --debug                if set, drop to IPython debug prompt on exception
#  -v, --verbose              verbosity (or -vv, or -vvv)
```

📈 Plot the data variables in captures contained in one sweep of a zarr archive, given a YAML plotting configuration specification:

```sh
plot-capture [OPTIONS] ZARR_PATH YAML_PATH

# Selected options:
#  -i, --interactive
#  --no-save          don't save the resulting plots
```

More detailed usage instructions for these tools can be discovered with the `--help` flag.

### Python module APIs
The API is organized into python packages:

#### `striqt.analysis`
Provides optimized and validated real-time signal analysis of complex-valued IQ baseband:
* Power spectral density and spectrogram evaluation
* Power detectors in various time-domain representation
* Empirical statistical distributions
* Cellular cyclic prefix and synchronization correlators
* Extensible with custom measurements registered beside the built-in ones

Interoperability within the modern python data ecosystem:
* Interchangeable across CPU (numpy) and GPU (cupy) in 32-bit floating point
* Package as multi-dimensional [`xarray.Dataset` objects](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) 
    - Detailed coordinates, units, and metadata across all axes
* Load and save [`zarr`](https://zarr.dev/) dataset archive format for easy aggregation and dissemination local or cloud storage

#### `striqt.sensor`
Implements batch IQ data acquisition and resampling oriented toward test and measurement
* Input power level calibration based on the Y-factor technique
* Support for exact rational Fourier resampling on CPU/GPU
* Multi-threaded acquisition, analysis, and archival to maximize throughput
* Software-defined radio interoperability via `SoapySDR`
* Acquisition and analysis input specification schema (and yaml input decode support)

#### `striqt.figures`
Implements visualization for the data variables produced by `striqt.analysis`
* Provides a plot function for each type of data variable
* Uses the labeled coordinates and metadata to display proper units
* Generates plots with `matplotlib` or `plotly` through one backend protocol

### Installation
The following assume access to the open internet.

#### Option 1: pixi environment
The repository carries a [`pixi`](https://pixi.sh) manifest with the conda-forge dependencies, including `SoapySDR`, and a `cupy` environment for CUDA GPUs:

1. Install `pixi` and clone this repository
2. In the repository, run `pixi install -e py315` (CPU) or `pixi install -e cupy` (CUDA GPU, linux or windows)
3. Run the tools through the environment, for example `pixi run -e py315 sensor-sweep --help`

#### Option 2: pip installation
An environment for post-analysis, plotting, and testing can be installed from the repository:

```sh
pip install "striqt @ git+https://github.com/usnistgov/striqt"
```

A lockfile (through `uv`, `pixi`, or `pipenv`) pins the dependency versions of an install; a container or virtual environment isolates it from other packages.

> **_NOTE:_** `SoapySDR` is required to acquire radio captures. It is distributed on conda-forge, not on `pypi`.

### Development status
`striqt` is in early beta.
* The API in the base of each package is expected not to change, but internals (`.lib`, etc.) may change without warning
* The yaml schemas of `striqt.sensor` and `striqt.figures`, and the CLI options, may still change
* The data variable, coordinate, and metadata field names in `.zarr` files follow the names in the yaml schemas

## Documentation
* The [documentation site](https://pages.nist.gov/striqt/) renders the command-line and API reference from the docstrings
* Worked sweep specifications are the YAML files under [tests/sensor/sweeps](https://github.com/usnistgov/striqt/tree/main/tests/sensor/sweeps)
<!-- 
### See Also
* [Validation and calibration with hardware](https://github.com/usnistgov/striqt-tests) -->
