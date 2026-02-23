`striqt` is a set of open-source python packages for batched real-time signal analysis on CPU/GPU and rapid prototyping of bespoke radio environment sensors.

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
Provides validated real-time signal analysis of complex-valued ``IQ'' baseband:
* Power spectral density and spectrogram evaluation
* Power detectors in various time-domain representation
* Empirical statistical distributions
* Cellular cyclic prefix and synchronization correlators
* Extensible with custom analyses based on custom

Interoperability within the modern python data ecosystem:
* Interchangable across CPU (numpy) or GPU (cupy) at full-precision floating point
* Fast `numba` numerical kernels for speed and portability across operating systems
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
Implements visualization for in `striqt.analysis`
* Provides a plot function for each type of data variable
* Uses the labeled coordinates and metadata to display proper units
* Generates plots in `matplotlib` for publication-ready in selectable output formats

### Installation
The following assume access to the open internet.

#### Option 1: Conda Environment
Installation with radio hardware and GPU support is provided via conda environments. Several variants of a `striqt` environment are provided here, targeted at different host computing environments.

1. Ensure that `conda` is installed (or `mamba`/`micromamba`, substituted in what follows)
2. Clone this repository
3. Download a predefined environment based on needed capabilities:
    - [`cpu.yml`](https://github.com/usnistgov/striqt/blob/main/environments/cpu.yml): acquire radio data, perform CPU-only IQ signal analysis, load/save `zarr` archives, and plot data
    - [`gpu-cpu.yml`](https://github.com/usnistgov/striqt/blob/main/environments/gpu-cpu.yml): support the above, and analyze IQ on CUDA GPUs
      
4. Create the chosen environment:
    ```sh
        conda env create -f [path-to-environment-here.yml]
    ```
4. To activate the environment, select the `striqt` conda environment in your IDE, or run 
    ```sh
        conda activate striqt
    ```

#### Option 2: pip installation
A limited environment that supports the latest APIs and CLIs for post-analysis, plotting, testing, etc. can be installed via the python package index (`pypi`):

```sh
pip install "striqt @ git+https://github.com/usnistgov/striqt"
```

It is encouraged to use a lockfile (through `pipenv`, `uv`, `pixi`, etc. in lieu of `pip` above) to enforce a fully reproducable install.

> **_NOTE:_** In order to create an environment that is reproducible environment and to isolate the install from dependency versioning conflicts, it is _highly_ recommended to install `striqt`` into a container or virtual environment.

> **_NOTE:_** `SoapySDR` is required to to acquire radio captures, but it is not distributed on `pypi`

### Development status
`striqt` is in early beta. The pace of change has slowed
* The API in the base of each module is expected not to change, but internals (`.lib`, etc.) may change without warning
* Incompatible changes to yaml schemas in `striqt.sensor` and `striqt.figure` (and the CLIs) may still change slightly
* The data variable, coordinate, and metadata field names in `.zarr` files follow the names in the yaml schemas

## Documentation
In keeping with the early beta development status of this codebase, documentation is limited.
* [reference](https://github.com/usnistgov/striqt/blob/main/doc/reference-sweep.yaml) for the `yaml` configuration files that drive `edge_sensor`
* Docstrings in the source code are the most up-to-date API documentation
* A few examples are located in [tests](https://github.com/usnistgov/striqt/tree/main/tests) and [notebooks](https://github.com/usnistgov/striqt/tree/main/notebooks). Some of these may not be up to date.
<!-- 
### See Also
* [Validation and calibration with hardware](https://github.com/usnistgov/striqt-tests) -->
