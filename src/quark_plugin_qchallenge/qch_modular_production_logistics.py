from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from qchallenge_framework import model_classes


@dataclass
class MPLProblem(Core):
    """Modular Production Logistics problem module from the QCHALLenge Project."""

    agv: int = 2
    jobs: int = 4
    transport_time: int = 2
    machines: int = 2
    processing_times: F

    seed: int = None


    @override
    def preprocess(self, data: InterfaceType) -> Result:
        N_A = (self.jobs + 1) // 2
        N_B = self.jobs // 2
        if self.processing_times is None:
            processing_times = {
                "Jobs_A": (7, 5),
                "Jobs_B": (4, 6, 4)
            }
        
        time_A = sum(processing_times["Jobs_A"])
        time_B = sum(processing_times["Jobs_B"])

        work_time = (N_A * time_A + N_B * time_B)*self.transport_time
        estimated_time = work_time / self.agv
        
        # timehorizont with buffer
        #TODO auch als feld
        T = int(estimated_time * 1.25 + 10)
        
        # Timelimit
        total_jobs = N_A + N_B
        if total_jobs <= 6:
            timelimit = 300
        elif total_jobs <= 10:
            timelimit = 600  
        elif total_jobs <= 14:
            timelimit = 1200
        else:
            timelimit = 1800
            
        params = {
            "N_A": N_A,
            "N_B": N_B,
            "M": self.machines,    # Machines    
            "R": self.agv,
            "t_r": self.transport_time,         # Fix: AGV Transport-Zeit
            "T": T,
            "processing_times": processing_times,
            "timelimit": timelimit
        }

        data = model_classes["MPL"]["data"]
        self.problem = data.create_problem(m=self.machines, j=self.jobs, seed=self.seed)

        milp = model_classes["PAS"]["cplex_model"]
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

        evaluator = model_classes["PAS"]["evaluation"](self.problem, original_solution)
        
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