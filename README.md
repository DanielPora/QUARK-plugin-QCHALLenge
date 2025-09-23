# QUARK-plugin-QCHALLenge

This module provides usecases from the QCHALLenge project[^1] implemented by Aqarios[^2].
These usecases consists of industry relevant problems which are formulated as mixed integer linear programs (MILP).
All of them introduce constraints in different magnitudes and will be therefore challenging for QUBO solvers. 
In addition to custom efficient qubo formulations for some usecases a general qubo mapping is provided.

## Usecases

| Module                            | Upstream Interface | Downstream Interface               |
|-----------------------------------|--------------------|------------------------------------|
| `qch_sensor_positioning`          | None               | `quark.interface_types.other (LP)` |
| `qch_sensor_positioning_qubo`     | None               | `quark.interface_types.qubo`       |
| `qch_production_assignment`       | None               | `quark.interface_types.other (LP)` |
| `qch_production_assignment_qubo`  | None               | `quark.interface_types.qubo`       |
| `qch_train_routing`               | None               | `quark.interface_types.other (LP)` |
| `qch_sensor_auto_carrier_loading` | None               | `quark.interface_types.other (LP)` |
| `qch_modular_production_logistics`| None               | `quark.interface_types.other (LP)` |


## Mapping

| Module                           | Upstream Interface                 | Downstream Interface           |
|----------------------------------|------------------------------------|--------------------------------|
| `qch_qubo_map`                   | `quark.interface_types.other (LP)` | `quark.interface_types.qubo`   |

[^1]: QCHALLenge project information: https://qarlab.de/qchallenge/ and <br>
https://www.digitale-technologien.de/DT/Navigation/DE/ProgrammeProjekte/AktuelleTechnologieprogramme/Quanten_Computing/Projekte/QCHALLenge/qchallenge.html
[^2]: https://www.aqarios.com