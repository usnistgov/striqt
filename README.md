This is a base library and collection of scripts oriented toward GPU-accelerated real-time RF environmental monitoring with support for software-defined radios. The lowest-level routines are implemented separetely in [iqwaveform](https://github.com/dgkuester/iqwaveform), which allow interchangable operation on `numpy` or `cupy` arrays to transparently evaluate on either a CPU or CUDA GPU.

### Development status
This is alpha software. The API may change frequently without warning.

### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure creates a python environment tailored based on hardware. This includes an editable install of the internal python modules implemented in `src/`. The only supported version of python is 3.9, in order to accommodate broad CUDA platform support. 

1. Ensure that `conda` is installed (or `mamba`/`micromamba`, substituted in what follows)
2. Clone this repository
3. Select a predefined environment based on use-case and hardware:
    - `environments/post-analysis-cpu.yml`: Cross-platform analysis
    - `environments/post-analysis-cpu-cuda.yml`: Analysis with added CUDA-specific GPU acceleration
    - `environments/edge-airt-cuda.yml`: Real-time edge sensing with AirT/AirStack radios
4. Create:
    ```sh
        conda env create -f <name.yml>
    ```
4. Activate:
    - For IDEs, select the `gpu-spectrum-sensor` virtual environment 
    - For a command line, `conda activate gpu-spectrum-sensor`