from typing import List, Dict, Any, Tuple
import numpy as np

def bqm_to_qubo_matrix(bqm) -> Tuple[np.ndarray, List[Any], Dict[int, Any]]:
    variables = list(bqm.variables)
    n = len(variables)
    
    var_to_index = {var: i for i, var in enumerate(variables)}
    index_to_var = {i: var for var, i in var_to_index.items()}
    
    qubo_matrix = np.zeros((n, n))

    for var, bias in bqm.linear.items():
        i = var_to_index[var]
        qubo_matrix[i, i] = bias

    for (var1, var2), bias in bqm.quadratic.items():
        i = var_to_index[var1]
        j = var_to_index[var2]
        if i <= j:
            qubo_matrix[i, j] = bias
        else:
            qubo_matrix[j, i] = bias
    
    return qubo_matrix, variables, index_to_var