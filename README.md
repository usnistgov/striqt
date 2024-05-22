This is a base library and collection of scripts oriented toward CUDA-accelerated RF monitoring with software-defined radios.

### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure needs to be followed to create an environment that across operating systems and including the software-defined radios for deployment. This creates a virtual environment with the only supported version of python (3.9), a mixture of conda and pip package dependencies, and an editable install that allows imports from subdirectories of `src/`. 

In order to get started:
1. Ensure that `conda` is installed (or substitute `mamba`/`micromamba` if preferred)
2. Clone this repository
3. Install one an environment depending on your available GPU resources:

    - To support CUDA GPU acceleration (requires hardware GPU support):

        ```sh
            conda env create -f environments/cuda.yml
        ```

    - For cross-platform CPU-only support:
        Otherwise:
        ```sh
            conda env create -f environments/cpu-only.yml
        ```

    - Apple Metal GPUs support is not yet functional, pending more features in `mlx`. Still, an environment is provided for development support:
        ```sh
            conda env create -f environments/mlx.yml

4. Activate the environment by selecting the `spectrum-sensor-edge-analysis` virtual environment in your IDE, or run the following to use in a command line environment:

    ```sh
        conda activate spectrum-sensor-edge-analysis
    ```