from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other, Qubo

from qchallenge_framework import model_classes

@dataclass
class PASProblemQubo(Core):
    """
        The production assignment problem (PAS) is a combinatorial optimization problem formulated within
        QCHALLenge. It is focused on reducing the manufacturing time of products or "jobs" while at the same time
        maximizing some value of the items produced, this can be profit or quality of the product, depending on the
        real-world application. The jobs are to be done in a fixed number of machines. The total manufacturing time
        is composed of the time it takes to manufacture each product and the setup process each machine undergoes
        between jobs.
        
        Parameters:
            machines (int): Number of machines used. Default: 2
            
            jobs (int): Number of jobs to perform. Default: 4
                
            seed (int): Seed for the RNG. Default: None
    """

    machines: int = 2
    jobs: int = 4
    seed: int = None

  
    @override
    def preprocess(self, data: InterfaceType) -> Result:
        data = model_classes["PAS"]["data"]
  
        self.problem = data.create_problem(m=self.machines, j=self.jobs, seed=self.seed)
 
        self.model = model_classes["PAS"]["qubo_model"](self.problem)
        qubo = self.model.get_model()
        return Data(Qubo.from_matrix(qubo))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        sample = self._solution_to_sample(data.data)
        solution_dict = self.model.decode_solution_2(sample)
        evaluator = model_classes["PAS"]["evaluation"](self.problem, solution_dict)

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
    
    def _solution_to_sample(self, solution_dict) -> list:
        x_vars = {k: v for k, v in solution_dict.items() if k.startswith('x_')}
        sorted_vars = sorted(x_vars.items(), key=lambda x: int(x[0].split('_')[1]))
        
        return [int(round(value)) for _, value in sorted_vars]
    
    
