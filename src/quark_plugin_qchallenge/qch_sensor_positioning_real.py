from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data
from quark.interface_types import InterfaceType, Other

from qchallenge_framework import model_classes

@dataclass
class SpRealProblem(Core):
    """
    This is an example module following the recommended structure for a quark module.

    A module must have a preprocess and postprocess method, as required by the Core abstract base class.
    A module's interface is defined by the type of data parameter those methods receive and return, dictating which other modules it can be connected to.
    Types defining interfaces should be chosen form QUARKs predefined set of types to ensure compatibility with other modules. TODO: insert link
    """
    lidar_density : float = 0.1
    street_point_density : float = 0.2

    @override
    def preprocess(self, data: InterfaceType) -> Result:
        sp_data = model_classes["SP"]["data"]
  
        problem = sp_data.create_problem_from_glb_file(lidar_density=self.lidar_density, street_point_density=self.street_point_density)
 
        sp_milp = model_classes["SP"]["cplex_model"]
        model = sp_milp(problem).model
        for idx, var in enumerate(model.iter_variables()):
            var.name = f"x_{idx}"
        lp_string = model.export_as_lp_string()
        return Data(Other(lp_string))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        print("real lp sol: ", data.data)
        return Data(Other(None))
