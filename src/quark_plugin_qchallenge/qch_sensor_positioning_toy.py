from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from qchallenge_framework import model_classes


@dataclass
class SpToyProblem(Core):
    """SP toy problem module with variable name mapping for LP export/import."""
    
    num_cols: int = 5
    version: int = 3
    max_radius: float = 2.5
    hor_basic_distance: float = 1.0
    vert_basic_dist: float = 2.0
    seed: int = 1


    @override
    def preprocess(self, data: InterfaceType) -> Result:
        sp_data = model_classes["SP"]["data"]
        self.problem = sp_data.create_problem(
            version=self.version, num_cols=self.num_cols, max_radius=self.max_radius,
            hor_basic_distance=self.hor_basic_distance, seed=self.seed)

        sp_milp = model_classes["SP"]["cplex_model"]
        lp_model = sp_milp(self.problem).model
        
        self.var_mapping = {}
        for idx, var in enumerate(lp_model.iter_variables()):
            new_name = f"x_{idx}"
            self.var_mapping[new_name] = var.name
            var.name = new_name
        
        return Data(Other(lp_model.export_as_lp_string()))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        original_solution = {
            self.var_mapping.get(k, k): v for k, v in data.data.items()
        }

        evaluator = model_classes["SP"]["evaluation"](self.problem, original_solution)
        missed_streetpoints = len(evaluator.check_solution()["missing_achievable_coverage"])
        if missed_streetpoints:
            return Failed(f"Invalid solution: {missed_streetpoints} street points not covered.")
        else:    
            return Data(Other(evaluator.get_objective()))