`striqt` provides a python-based toolset for rapid prototyping of
bespoke real-time signal analyzers (RTSAs).

## Basic Usage

### Command line
Once a `striqt` environment is installed and activated, the following scripts are installed into the environment `PATH`, so they can be run from any working directory.

* `sensor-sweep`: Acquire and analyze a capture sequence (sweep) according to [a YAML input file specification](https://github.com/usnistgov/striqt/blob/main/doc/reference-sweep.yaml).
  Output datasets are serialized to `zarr` with `xarray`.
  This can run locally or on a remote host (`-r` argument) running `serve-sensor`.

* `serve-sensor`: Serve access to local compute and radio resources for `sensor-sweep` on remote clients.

Detailed usage instructions for each can be discovered with the `--help` flag.

### Python module APIs
This is alpha software. The API may still change without warning.

The API is organized into two python modules that are importable as :

`striqt.analysis`
* Validated routines to perform real-time signal analysis (RTSA) from complex-valued baseband
    - Spectrum and spectrogram evaluation
    - Empirical statistical distributions
    - Cellular cyclic prefix and synchronization correlators
* Packaging as [xarray Dataset objects](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) for easy-to-explore data management
* Archive datasets to `zarr` for local or cloud storage support
* Optimized CPU or CUDA GPU compute using full-precision float

`striqt.sensor`
* Acquire data from sensor hardware according to a configuration file
* Concurrent acquisition, analysis, and archival
* Automated sensor calibration based on the Y-factor technique

## Installation
The following options require that the host has internet access in order to download dependencies. Installs via local package indexes may require customization that has not been tested.

### Option 1: Conda Environment
Installation with radio hardware and GPU support is provided via conda environments. Several variants of a `striqt` environment are provided here, targeted at different host computing environments.

1. Ensure that `conda` is installed (or `mamba`/`micromamba`, substituted in what follows)
2. Clone this repository
3. Select a predefined environment based on use-case and hardware:
    - `environments/cpu.yml`: Analyze pre-recorded IQ or run remote control (cross-platform, CPU only)
    - `environments/gpu-cpu.yml`: Analyze pre-recorded IQ or run remote control (cross-platform, CPU or CUDA GPU)
    - `environments/edge-airt.yml`: Signal acquisition and analysis to run on AirT/AirStack radios
4. Create the chosen environment:
    ```sh
        conda env create -f <path-to-environment-here.yml>
    ```
4. Activate:
    - IDE: select the `striqt` virtual environment 
    - Command line: `conda activate striqt`

> **_NOTE:_**  The environment operates on an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) of the modules and command line tools. As a result, if the location of the cloned source code repository is moved, the conda environment needs to be removed and built again according to the instructions above.

### Option 2: pip installation
The dependencies, APIs, and CLIs can be installed without radio hardware or GPU support (for post-analysis, plotting, testing, etc.) via `pip install`. In order to avoid conflicts with other projects, the recommended practice for this is to install into a python virtual environment.

1. Clone this repository
2. `pip install <path-to-repository>`

## Documentation
In keeping with the alpha development status of this codebase, documentation is limited.
* [reference](https://github.com/usnistgov/striqt/blob/main/doc/reference-sweep.yaml) for the `yaml` configuration files that drive `edge_sensor`
* Docstrings in the source code are the most up-to-date API documentation
* A few examples are located in [tests](https://github.com/usnistgov/striqt/tree/main/tests) and [notebooks](https://github.com/usnistgov/striqt/tree/main/notebooks). Some of these may not be up to date.

### See Also
* [Validation and calibration with hardware](https://github.com/usnistgov/striqt-tests)
