from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other, Qubo

from framework.init import model_classes

@dataclass
class SPProblemQubo(Core):
    """
    Custom QUBO formulation for the Sensor Positioning problem from the QCHALLenge project.

    This module addresses a sensor positioning use case where street points need to be 
    covered by lidars that can be placed at potential lidar positions. There are two 
    different problem structures: toy and real-world.
    
    Toy Structure:
        A grid-based layout with columns and rows where lidars are positioned to cover
        street points in a structured pattern.
    
    Real-world Structure:
        A factory hall environment with roadways and obstacles where street point and
        lidar densities determine the accuracy for the coverage.
    
    Parameters:
        Toy problem
                num_cols (int): Number of columns for the toy problem grid structure. Default: 5
                
                version (int): Problem layout version for toy structure. Default: 3
                    - Version 1: One row of lidars above one row of street points
                    - Version 2: One row of lidars above two rows of street points  
                    - Version 3: Three rows of street points with one row of lidars above and below
                    
                max_radius (float): Coverage range/radius of the lidars in distance units. Default: 2.5
                
                hor_basic_distance (float): Horizontal spacing/width between columns in the grid. Default: 1.0
                
                vert_basic_dist (float): Vertical spacing/width between rows in the grid. Default: 2.0
                
                real_world_problem (bool): Toggle between toy grid structure (False) and 
                    real-world factory environment (True). Default: False

        Real-world problem

            lidar_density (float): Density of potential lidar placement positions in the 
                real-world problem structure. Only relevant when real_world_problem=True. Default: 0.1
                
            street_point_density (float): Density of street points to be covered in the 
                real-world problem structure. Only relevant when real_world_problem=True. Default: 0.1
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
