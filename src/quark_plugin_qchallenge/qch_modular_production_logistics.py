from dataclasses import dataclass, field
from typing import override
import os
import tempfile

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from qchallenge_framework import model_classes


@dataclass
class MPLProblem(Core):
    """
    Modular Production Logistics problem module from the QCHALLenge Project.
    
    Parameters:
            agv (int): Number of agvs used. Default: 2
            
            jobs (int): Number of jobs to perform, equally split into A and B jobs. Default: 4
            
            transport_time (int): 

            processing_times (dict[int]):

            time_horizont (int): Discretized maximum time that is allowed. Will limit the search space, but set too low might lead to infeasible problems. 
                
            seed (int): Seed for the RNG. Default: None
    """

    agv: int = 2
    jobs: int = 2
    processing_times: dict = field(
        default_factory=lambda: {"Jobs_A": (1, 1), "Jobs_B": (1, 1, 1)}
    )
    time_horizont: int = None
    seed: int = None

    @override
    def preprocess(self, data: InterfaceType) -> Result:
        N_A = (self.jobs + 1) // 2
        N_B = self.jobs // 2
                
        # timehorizont with buffer
        if self.time_horizont is None:
            time_A = sum(self.processing_times["Jobs_A"])
            time_B = sum(self.processing_times["Jobs_B"])

            work_time = (N_A * time_A + N_B * time_B)*2
            estimated_time = work_time / self.agv

            T = int(estimated_time * 1.25)
        else:
            T = self.time_horizont
        
        params = {
            "N_A": N_A,
            "N_B": N_B,
            "R": self.agv,
            "T": T,
            "processing_times": self.processing_times,
            "seed": self.seed,
        }

        data = model_classes["MPL"]["data"]
        self.problem = data.create_problem(**params)

        milp = model_classes["MPL"]["gurobinowaitnooverlap_model"]
        self.gurobi_instance = milp(self.problem)

        self.var_mapping = {}
        gurobi_model = self.gurobi_instance.model

        # for idx, var in enumerate(gurobi_model.getVars()):
        #     original_name = var.VarName
        #     new_name = f"x_{idx}"
        #     self.var_mapping[new_name] = original_name
        #     var.VarName = new_name
        # gurobi_model.update()
        lp_string = self.get_lp_string_from_gurobi(gurobi_model)
        print("lp string", lp_string)
        return Data(Other(lp_string))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        original_solution = {
            self.var_mapping.get(k, k): v for k, v in data.data.items()
        }

        evaluator = model_classes["MPL"]["evaluation"](self.problem, original_solution)
        
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
        
    def get_lp_string_from_gurobi(self, gurobi_model):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            gurobi_model.write(temp_filename)
            with open(temp_filename, 'r', encoding='utf-8') as f:
                lp_string = f.read()
            
            return lp_string
            
        finally:
            try:
                os.remove(temp_filename)
            except OSError as e:
                print(f"Warning: Could not delete temporary file {temp_filename}: {e}")
                pass