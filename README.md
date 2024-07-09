This is a collection of python scripts packaged with underlying libraries for GPU-accelerated real-time RF environmental monitoring with support for software-defined radios. Baseband signal processing may be performed interchangeably on either CPU or CUDA GPUs.

### Tooling ecosystem and compatibility
Python abstraction libraries like `SoapySDR` and `Array API` are used to minimize development time across different radio and compute hardware.
* Low-level DSP is based on [iqwaveform](https://github.com/dgkuester/iqwaveform)
* The [python array API](https://data-apis.org/array-api/latest/) is used to access `numpy` or `cupy` interchangeably
* Interactive datasets are built and serialized to disk with [xarray](https://docs.xarray.dev/en/stable/)

### Development status
This is alpha software. The API changes frequently without warning.

### Environment setup
The source code layout is oriented toward execution of notebooks or scripts in conda environments.

The following setup procedure creates a python environment tailored based on hardware. This includes an editable install of the internal python modules implemented in `src/`. The only supported version of python is 3.9, in order to accommodate broad CUDA platform support. 

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
    - For IDEs, select the `flex-spectrum-sensor` virtual environment 
    - For a command line, `conda activate flex-spectrum-sensor`
  
### Usage
The following scripts can be run from any of the `environments/edge-*` environments, which install all scripts into the command line path. 
* `edge-sensor-sweep`: evaluate baseband captures of wireless channels across multiple or repeated frequencies, gains, etc
