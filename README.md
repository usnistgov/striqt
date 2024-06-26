This is a base library and collection of scripts oriented toward CUDA-accelerated RF monitoring with software-defined radios.

### Environment setup
The source code layout is oriented toward execution of notebooks/scripts in conda environments.

The following setup procedure needs to be followed to create an environment that across operating systems and including the software-defined radios for deployment. This creates a virtual environment with the only supported version of python (3.9), a mixture of conda and pip package dependencies, and an editable install that allows imports from subdirectories of `src/`. 

In order to get started:
1. Ensure that `conda` is installed (or substitute `mamba`/`micromamba` if preferred)
2. Clone this repository
3. Select and create a conda environment for your use case and hardware:
    ```sh
        conda env create -f environments/<name.yml>
    ```

    Replace `<name.yml>` with one of the following:

    - For post-analysis and testing:
        - `post-analysis-cpu.yml`: Cross-platform CPU analysis:
        - `post-analysis-cpu-cuda.yml`: Cross-platform CPU _and_ GPU acceleration with CUDA

    - For real-time sensor acquisition and edge analysis:
        - `edge-airt-cuda.yml`: Real-time edge sensor on AirT/AirStack SDR platform
            

4. Activate the environment by selecting the `gpu-spectrum-sensor` virtual environment in your IDE, or run the following to use in a command line environment:

    ```sh
        conda activate gpu-spectrum-sensor
    ```
