This is a collection of python modules and scripts for GPU-accelerated RF environmental monitoring and analysis with software-defined radios. Baseband signal processing may be performed interchangeably on either CPU or CUDA GPUs.

## Usage

### Environment installation
The source code layout is oriented toward execution of notebooks or scripts in conda environments. Several variants of a `flex-spectrum-sensor` environment are provided here, targeted at different host computing environments.

In order to accommodate broad support for legacy CUDA platforms, the required version of python is 3.9.

1. Ensure that `conda` is installed (or `mamba`/`micromamba`, substituted in what follows)
2. Clone this repository
3. Select a predefined environment based on use-case and hardware:
    - `environments/channel-analysis-cpu.yml`: Analyze pre-recorded IQ (CPU backend only)
    - `environments/channel-analysis-cpu-cuda.yml`: Analyze pre-recorded IQ (intercheangable CPU or CUDA backends)
    - `environments/edge-sensor-airt.yml`: Real-time sensing and analysis with AirT/AirStack radios
4. Create the chosen environment:
    ```sh
        conda env create -f <path-to-environment-here.yml>
    ```
4. Activate:
    - IDE: select the `flex-spectrum-sensor` virtual environment 
    - Command line: `conda activate flex-spectrum-sensor`


### Command line
Once a `flex-spectrum-sensor` environment is installed and activated, the following scripts are installed into the environment `PATH`, so they can be run from any working directory.

* `edge-sensor-sweep`: Acquire and analyze a capture sequence (sweep) according to a YAML input file specification.
  Output datasets are serialized to `zarr` with `xarray`.
  This can run locally or on a remote host (`-r` argument) running `edge-sensor-server`.

* `edge-sensor-server`: Serve access to local compute and radio resources for `edge-sensor-sweep` on remote clients.

Detailed usage instructions for each can be discovered with the `--help` flag.

### Module APIs
This is alpha software. The API may still change without warning, and only source-code level documentation is available for these modules.

The repository is organized into two python modules that are importable as [editable installs](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) within the `flex-spectrum-sensor` environment:

* `channel_analysis`: Methods for the analysis of an IQ recording. These use [iqwaveform](https://github.com/dgkuester/iqwaveform) and [the python array API](https://data-apis.org/array-api/latest/) for interchangeable CPU or CUDA GPU compute, depending on whether `numpy` or `cupy` objects are passed in. Results are packaged into [xarray Dataset objects](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).

* `edge_sensor`: Methods for swept acquisition and analysis of field data with software-defined radios.

These may be imported from any directory provided that the `flex-spectrum-sensor` environment is activated. They are imported directly from the file tree in the source code repository, so __if your copy of the source code repository is moved, the flex-spectrum-sensor environment should be removed and built again from scratch__.