from dataclasses import dataclass
from typing import override

from quark.core import Core, Result, Data, Failed
from quark.interface_types import InterfaceType, Other
from qchallenge_framework import model_classes


@dataclass
class SPProblem(Core):
    """SP toy problem module with variable name mapping for LP export/import.
    
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
    
    num_cols: int = 5
    version: int = 3
    max_radius: float = 2.5
    hor_basic_distance: float = 1.0
    vert_basic_dist: float = 2.0

    real_world_problem: bool = False
    
    lidar_density: float = 0.1
    street_point_density: float = 0.1

    @override
    def preprocess(self, data: InterfaceType) -> Result:
        sp_data = model_classes["SP"]["data"]
        if self.real_world_problem:
            self.problem = sp_data.create_problem_from_glb_file(lidar_density=self.lidar_density, street_point_density=self.street_point_density)
        else:
            self.problem = sp_data.create_problem(
                version=self.version, num_cols=self.num_cols, max_radius=self.max_radius,
                hor_basic_distance=self.hor_basic_distance)

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