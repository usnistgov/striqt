This is a base library and collection of scripts oriented toward CUDA-accelerated RF monitoring with software-defined radios.

### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure needs to be followed to create an environment that across operating systems and including the software-defined radios for deployment. This creates a virtual environment with the only supported version of python (3.9), a mixture of conda and pip package dependencies, and an editable install that allows imports from subdirectories of `src/`. 

In order to get started:
1. Ensure that `conda` is installed (or substitute `mamba`/`micromamba` if preferred)
2. Clone this repository
3. Create a conda environment to suit your use case and hardware:
    - For post-analysis and testing:
        - Cross-platform CPU analysis:
            ```sh
                conda env create -f environments/analysis-cpu.yml
            ```
        - Cross-platform CPU, plus GPU acceleration with CUDA:
            ```sh
                conda env create -f environments/analysis-cpu-cuda.yml
            ```

    - For real-time SDR acquisition and analysis:
        - Jetson TX2/AirStack with CUDA GPU acceleration:
            ```sh
                conda env create -f environments/cuda.yml
            ```

4. Activate the environment by selecting the `gpu-spectrum-sensor` virtual environment in your IDE, or run the following to use in a command line environment:

    ```sh
        conda activate gpu-spectrum-sensor
    ```
