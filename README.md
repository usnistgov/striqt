This is a collection of python scripts packaged with underlying libraries for GPU-accelerated real-time RF environmental monitoring with support for software-defined radios. Baseband signal processing may be performed interchangeably on either CPU or CUDA GPUs.

## Usage

### Environment installation
The source code layout is oriented toward execution of notebooks or scripts in conda environments. Several variants of a `flex-spectrum-sensor` environment are provided here, targeted at different host computing environments.

Each environment incorporates an editable install of the internal python modules implemented in `src/`, and installs . The only supported version of python is 3.9, in order to accommodate broad CUDA platform support. 

1. Ensure that `conda` is installed (or `mamba`/`micromamba`, substituted in what follows)
2. Clone this repository
3. Select a predefined environment based on use-case and hardware:
    - `environments/post-analysis-cpu.yml`: Cross-platform analysis
    - `environments/post-analysis-cpu-cuda.yml`: Analysis with added CUDA-specific GPU acceleration
    - `environments/edge-airt-cuda.yml`: Real-time edge sensing with AirT/AirStack radios
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

Instructions for the use of these can be found with the `--help` flag. 

### Module APIs
This is alpha software. The API may still change without warning, and only source-code level documentation is available for these modules.

The repository is organized into two python modules that are importable as [editable installs](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) within the `flex-spectrum-sensor` environment:

* `channel_analysis`: Methods for the analysis of an IQ recording. These use [iqwaveform](https://github.com/dgkuester/iqwaveform) and [the python array API](https://data-apis.org/array-api/latest/) for interchangeable CPU or CUDA GPU compute, depending on whether `numpy` or `cupy` objects are passed in. Results are packaged into [xarray Dataset objects](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).

* `edge_sensor`: Methods for swept acquisition and analysis of field data with software-defined radios.

These may be imported from any directory provided that the `flex-spectrum-sensor` environment is activated. They are imported directly from the file tree in the source code repository, so __if your copy of the source code repository is moved, the flex-spectrum-sensor environment should be removed and built again from scratch__.