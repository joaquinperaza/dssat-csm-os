# I/O

The `dssat.io` package provides JSON readers (used by `Simulation.from_json`) and two-way converters between legacy DSSAT fixed-format files and JSON.

## JSON Readers

::: dssat.io.readers

---

## Converters

### Weather — `.WTH` ↔ JSON

::: dssat.io.converters.wth
    options:
      members:
        - read_wth
        - write_wth
        - wth_to_json_file
        - json_to_wth_file

### Soil — `.SOL` ↔ JSON

::: dssat.io.converters.sol
    options:
      members:
        - read_sol
        - read_sol_profile
        - write_sol
        - sol_to_json_file
        - json_to_sol_file

### Experiment X-file ↔ JSON

::: dssat.io.converters.xfile
    options:
      members:
        - read_xfile
        - write_xfile
        - xfile_to_json_file
        - json_to_xfile
