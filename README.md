This is a base library and collection of scripts oriented toward GPU-accelerated real-time RF environmental monitoring with support for software-defined radios.

### Development status
This is alpha software. The API may change frequently without warning.

### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure creates a python environment tailored based on hardware. This includes an editable install of the internal python modules implemented in `src/`. The only supported version of python is 3.9, in order to accommodate broad CUDA platform support. 

1. Ensure that `conda` is installed (or substitute `mamba`/`micromamba` in what follows)
2. Clone this repository
3. Select one an environment best suited to your use case and hardware:
    - `environments/post-analysis-cpu.yml`: Cross-platform analysis
    - `environments/post-analysis-cpu-cuda.yml`: Analysis with added CUDA-specific GPU acceleration
    - `environments/edge-airt-cuda.yml`: Real-time edge sensing with AirT/AirStack radios
4. Create the chosen conda environment:
    ```sh
        conda env create -f <name.yml>
    ```
4. Activate the environment by selecting the `gpu-spectrum-sensor` virtual environment in your IDE, or run the following to use in a command line environment:

    ```sh
        conda activate gpu-spectrum-sensor
    ```
