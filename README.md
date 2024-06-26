This is a base library and collection of scripts oriented toward CUDA-accelerated RF monitoring with software-defined radios.

### Development status


### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure creates a python environment tailored based on hardware. This includes an editable install of the internal python modules implemented in `src/`. The only supported version of python is 3.9, in order to accommodate broad CUDA platform support. 

In order to get started:
1. Ensure that `conda` is installed (or substitute `mamba`/`micromamba` in what follows)
2. Clone this repository
3. Select and create a conda environment for your use case and hardware:
    ```sh
        conda env create -f environments/<name.yml>
    ```

    Replace `<name.yml>` with one of the following:

    - For post-analysis and testing:
        - `post-analysis-cpu.yml`: Cross-platform CPU analysis
        - `post-analysis-cpu-cuda.yml`: Cross-platform CPU plus CUDA GPU acceleration

    - For real-time sensor acquisition and edge analysis:
        - `edge-airt-cuda.yml`: Real-time edge sensor on AirT/AirStack SDR platform
            

4. Activate the environment by selecting the `gpu-spectrum-sensor` virtual environment in your IDE, or run the following to use in a command line environment:

    ```sh
        conda activate gpu-spectrum-sensor
    ```
