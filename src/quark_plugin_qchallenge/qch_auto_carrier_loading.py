from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from qchallenge_framework import model_classes


@dataclass
class ACLProblem(Core):
    """Auto Carrier Loading (ACL) problem module from the QCHALLenge Project."""

    num_cars: int = 10
    num_trucks: int = 1
    seed:int = 1


    @override
    def preprocess(self, data: InterfaceType) -> Result:
        data = model_classes["ACL"]["data"]
        self.problem = data.create_problem(num_cars=self.num_cars, num_trucks=self.num_trucks, seed=self.seed)

        milp = model_classes["ACL"]["cplex_model"]
        lp_model = milp(self.problem).model
        
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

        evaluator = model_classes["ACL"]["evaluation"](self.problem, original_solution)
        violations = evaluator.check_solution()
        has_violations = any(constraint_violations for constraint_violations in violations.values())
        if has_violations:
            violation_summary = []
            for constraint_name, violation_list in violations.items():
                if violation_list:  # Only include constraints that were violated
                    violation_count = len(violation_list)
                    violation_summary.append(f"{constraint_name}: {violation_count} violations")
            
            violation_text = "; ".join(violation_summary)
            return Failed(f"Invalid solution! {violation_text}")
        else:    
            return Data(Other(evaluator.get_objective()))