# striqt.sensor

::: striqt.sensor
    options:
      members: false

## Running sweeps

Sweep execution against a bound sensor, and readers for sweep specs and calibration data.

::: striqt.sensor
    options:
      extra:
        module_docstring: false
      members:
        - iterate_sweep
        - open_resources
        - open_store
        - read_yaml_spec
        - read_json_spec
        - read_zarr_spec
        - read_calibration
