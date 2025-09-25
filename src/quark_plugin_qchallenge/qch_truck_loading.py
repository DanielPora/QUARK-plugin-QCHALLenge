from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from framework.init import model_classes


@dataclass
class TLProblem(Core):
    """
    Create a 2-D truck loading problem for fragile not stackable boxes with a rectangular base area. 
    A defined number of boxes are created with random lenght, width and weigth.
    Goal of the optimization problem is to utilize the available cargo space optimally 
    while respecting the weigth limit and dimensions of the truck.
 
    
    Parameters:
        num_boxes (int): Number of boxes to load. Default 5
            
        seed (int): Seed for the RNG. Default: 1
    """

    num_boxes: int = 5
    seed: int = 1


    @override
    def preprocess(self, data: InterfaceType) -> Result:
        data = model_classes["TL"]["data"]
        self.problem = data.create_problem(num_boxes=self.num_boxes, seed=self.seed)

        self.milp = model_classes["TL"]["cplex_model"](self.problem)
        self.lp_model = self.milp.model
        
        self.var_mapping = {}
        for idx, var in enumerate(self.lp_model.iter_variables()):
            new_name = f"x_{idx}"
            self.var_mapping[new_name] = var.name
            var.name = new_name
        return Data(Other(self.lp_model.export_as_lp_string()))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        original_solution = {}
        original_solution["solution"] = {
            self.var_mapping.get(k, k): v for k, v in data.data.items()
        }
        selected_boxes = self.milp.select_boxes(original_solution["solution"])
        original_solution["solution"]["solution"] = selected_boxes
        evaluator = model_classes["TL"]["evaluation"](self.problem, original_solution)

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