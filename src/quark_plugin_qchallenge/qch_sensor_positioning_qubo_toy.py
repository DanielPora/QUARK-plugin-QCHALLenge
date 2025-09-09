from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other, Qubo

from qchallenge_framework import model_classes

@dataclass
class SpToyProblemQubo(Core):
    """
    This is an example module following the recommended structure for a quark module.

    A module must have a preprocess and postprocess method, as required by the Core abstract base class.
    A module's interface is defined by the type of data parameter those methods receive and return, dictating which other modules it can be connected to.
    Types defining interfaces should be chosen form QUARKs predefined set of types to ensure compatibility with other modules. TODO: insert link
    """

    num_cols : int = 5
    version : int = 3
    max_radius : float = 2.5
    hor_basic_distance : float = 1.0
    vert_basic_dist : float = 2.
    seed : int = 1
  
    @override
    def preprocess(self, data: InterfaceType) -> Result:
        sp_data = model_classes["SP"]["data"]
  
        self.problem = sp_data.create_problem(version=self.version, num_cols=self.num_cols, max_radius=self.max_radius, 
                                             hor_basic_distance=self.hor_basic_distance, seed=self.seed)
 
        self.model = model_classes["SP"]["qubobinary_model"](self.problem)
        qubo = self.model.model
        
        return Data(Qubo.from_matrix(qubo))

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        sample = self._solution_to_sample(data.data)
        solution_dict = self.__inverter_solution(sample)
        print("toy lp spl: ", data.data)
        evaluator = model_classes["SP"]["evaluation"](self.problem, solution_dict)
        missed_streetpoints = len(evaluator.check_solution()["missing_achievable_coverage"])
        if missed_streetpoints:
            return Failed(f"Invalid solution: {missed_streetpoints} street points not covered.")
        else:    
            return Data(Other(evaluator.get_objective()))
    
    def _solution_to_sample(self, solution_dict) -> list:
        x_vars = {k: v for k, v in solution_dict.items() if k.startswith('x_')}
        sorted_vars = sorted(x_vars.items(), key=lambda x: int(x[0].split('_')[1]))
        
        return [int(round(value)) for _, value in sorted_vars]
    
    def __inverter_solution(self, sample):
        solution_dict = {
            f"x_{self.model.usedLidars[i][0]}_{self.model.usedLidars[i][1]}_{self.model.usedLidars[i][2]}_{self.model.usedLidars[i][3]}_{self.model.usedLidars[i][4]}": 
            float(sample[i])
            for i in range(len(self.model.usedLidars))
        }
        if self.model.reduced:
            for l in self.model.data.lidar1:
                solution_dict[f"x_{l[0]}_{l[1]}_{l[2]}_{l[3]}_{l[4]}"] = 1
            for l in self.model.data.lidar0:
                solution_dict[f"x_{l[0]}_{l[1]}_{l[2]}_{l[3]}_{l[4]}"] = 0
        return solution_dict
