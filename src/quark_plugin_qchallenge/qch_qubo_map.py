from dataclasses import dataclass
from typing import override, List, Dict, Any
import numpy as np

from quark.core import Core, Data, Result, Failed
from quark.interface_types import Other, Qubo
from .utils import bqm_to_qubo_matrix


import dimod


@dataclass
class QuboMap(Core):
    """A module for mapping an LP from Luna Usecases to a QUBO."""

    qubo_matrix: np.ndarray = None
    variable_order: List[Any] = None
    index_to_var: Dict[int, Any] = None

    @override
    def preprocess(self, data: Other) -> Result:
        self.cqm = dimod.lp.loads(data.data)
        self.bqm, self.inverter = dimod.cqm_to_bqm(self.cqm)
        # mapping vars to consistent variable names to make is solveable with luna quantum
        self.qubo_matrix, self.variable_order, self.index_to_var = bqm_to_qubo_matrix(self.bqm)
        
        return Data(Qubo.from_matrix(self.qubo_matrix))

    @override
    def postprocess(self, data: Other) -> Result:
        if data.data is None:
            return Failed("Solution is None.")
    
        bqm_solution = self.map_solution_to_bqm_variables(data.data)
        lp_solution = dict(self.inverter(bqm_solution))

        return Data(Other(lp_solution))
    

    
    def map_solution_to_bqm_variables(self, solution_dict: Dict[str, int]) -> Dict[Any, int]:
        """remapping the varibales to the original names used in the bqm"""
        bqm_solution = {}
        
        for i, original_var in enumerate(self.variable_order):
            qubo_var_name = f'x_{i}' 
            
            if qubo_var_name in solution_dict:
                bqm_solution[original_var] = solution_dict[qubo_var_name]
            else:
                print(f"Warning: Variable {qubo_var_name} not found in solution")
                bqm_solution[original_var] = 0 # using default 0, since the var did nothing for obj but need to be present
        
        return bqm_solution