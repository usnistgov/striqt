This is a python-based platform for experimentation with radio environmental analysis in real time with software-defined radios. Baseband signal processing may be performed interchangeably on either CPU or CUDA GPUs.

## Installation

### Conda Environment
The source code layout is oriented toward execution of notebooks or scripts in conda environments. Several variants of a `flex-spectrum-sensor` environment are provided here, targeted at different host computing environments.

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
    - IDE: select the `flex-spectrum-sensor` virtual environment 
    - Command line: `conda activate flex-spectrum-sensor`

> **_NOTE:_**  The environment operates on an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) of the modules and command line tools. This means you should not
move the `flex-spectrum-sensor` directory, otherwise the environment will be corrupted.  

### pip installation
The dependencies, APIs, and CLIs can be installed without radio hardware support (for post-analysis, testing, etc.) via `pip install`. In order to avoid conflicts with other projects, the recommended practice for this is to install into a python virtual environment.

Procedure:
1. Clone this repository
2. `pip install <path-to-repository>`

## Usage

### Command line
Once a `flex-spectrum-sensor` environment is installed and activated, the following scripts are installed into the environment `PATH`, so they can be run from any working directory.

* `edge-sensor-sweep`: Acquire and analyze a capture sequence (sweep) according to [a YAML input file specification](https://github.com/usnistgov/flex-spectrum-sensor/blob/main/doc/reference-sweep.yaml).
  Output datasets are serialized to `zarr` with `xarray`.
  This can run locally or on a remote host (`-r` argument) running `edge-sensor-server`.

* `edge-sensor-server`: Serve access to local compute and radio resources for `edge-sensor-sweep` on remote clients.

Detailed usage instructions for each can be discovered with the `--help` flag.

### Module APIs
This is alpha software. The API may still change without warning, and only source-code level documentation is available for these modules.

The repository is organized into two python modules that are importable as :

* `channel_analysis`: Methods for the analysis of an IQ recording. These use [iqwaveform](https://github.com/dgkuester/iqwaveform) and [the python array API](https://data-apis.org/array-api/latest/) for interchangeable CPU or CUDA GPU compute, depending on whether `numpy` or `cupy` objects are passed in. Results are packaged into [xarray Dataset objects](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).

* `edge_sensor`: Methods for swept acquisition and analysis of field data with software-defined radios.

These may be imported from any directory provided that the `flex-spectrum-sensor` environment is activated. They are imported directly from the file tree in the source code repository, so __if your copy of the source code repository is moved, the flex-spectrum-sensor environment should be removed and built again from scratch__.

### See Also
* [Validation and calibration with hardware](https://github.com/usnistgov/flex-spectrum-sensor-tests)
