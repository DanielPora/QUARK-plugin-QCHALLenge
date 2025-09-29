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

            T = int(estimated_time * 1.25 + 10)
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

        milp = model_classes["MPL"]["gurobiwaitoverlap_model"]
        self.gurobi_instance = milp(self.problem)

        gurobi_model = self.gurobi_instance.model
        gurobi_model.reset()  # This clears solution and cached info
        gurobi_model.update()

        # Remove all constraints with no variables
        constrs_to_remove = []
        for constr in gurobi_model.getConstrs():
            if gurobi_model.getRow(constr).size() == 0:
                constrs_to_remove.append(constr)

        if constrs_to_remove:
            print(f"Removing {len(constrs_to_remove)} empty constraints")
            gurobi_model.remove(constrs_to_remove)
            gurobi_model.update()

        # Remove redundant constraints
        redundant_constrs = self.find_and_remove_redundant_constraints(gurobi_model)
        if redundant_constrs:
            print(f"Removed {len(redundant_constrs)} redundant constraints")


        lp_string = self.get_lp_string_from_gurobi(gurobi_model)
        lp_string = self.rename_vars_in_lp_string(gurobi_model, lp_string)
        # if len(gurobi_model.getQConstrs())>0:
        #     print("quad const: ", gurobi_model.getQConstrs())
        # else:
        #     print("lp string", lp_string)
        
        scip_solve(lp_string, filename="lp_string.lp")
        return Data(Other(lp_string))
    
    def rename_vars_in_lp_string(self, gurobi_model, lp_string):
        mapping = {}
        vars = gurobi_model.getVars()

        for i, var in enumerate(vars):
            old = var.VarName
            new = f"x_{i}"
            mapping[new] = old  # reverse mapping for later
            lp_string = lp_string.replace(old, new)
        
        self.var_mapping = {v: k for k, v in mapping.items()}  # for postprocess
        return lp_string
    

    def find_and_remove_redundant_constraints(self, gurobi_model):
        """
        Find and remove redundant constraints from the Gurobi model.
        
        Returns:
            list: List of removed constraints
        """
        
        constraints = gurobi_model.getConstrs()
        constraint_signatures = {}
        redundant_constrs = []
        
        for constr in constraints:
            # Get the constraint row (linear expression)
            row = gurobi_model.getRow(constr)
            
            # Create a signature for the constraint
            # This includes: coefficients, variables, sense, and RHS
            var_coeffs = {}
            for i in range(row.size()):
                var = row.getVar(i)
                coeff = row.getCoeff(i)
                var_coeffs[var.VarName] = coeff
            
            # Create a hashable signature
            # Sort by variable name to ensure consistent ordering
            sorted_vars = tuple(sorted(var_coeffs.items()))
            sense = constr.Sense
            rhs = constr.RHS
            
            signature = (sorted_vars, sense, rhs)
            
            if signature in constraint_signatures:
                # This is a redundant constraint
                redundant_constrs.append(constr)
            else:
                # This is the first occurrence of this constraint
                constraint_signatures[signature] = constr
        
        # Remove redundant constraints
        if redundant_constrs:
            gurobi_model.remove(redundant_constrs)
            gurobi_model.update()
        
        return redundant_constrs

    @override
    def postprocess(self, data: InterfaceType) -> Result:
        renamed_solution = {}
        for k, v in data.data.items():
            original_name = self.var_mapping.get(k, k)
            renamed_solution[original_name] = v

        evaluator = model_classes["MPL"]["evaluation"](self.problem, renamed_solution)
        
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
            


def scip_solve(lp_string, filename: str = None):
    from pyscipopt import Model
    if filename is None:
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False)
        tmpfile.write(lp_string)
        tmpfile.flush()
        tmpfile_name = tmpfile.name
        delete_file = True
    else:
        with open(filename, 'w') as f:
            f.write(lp_string)
        tmpfile_name = filename
        delete_file = False

    try:
        model = Model()
        model.readProblem(tmpfile_name)
        model.optimize()

        status = model.getStatus()
        sol = model.getBestSol()
        if sol is None:
            solution = None
        else:
            solution = {var.name: model.getSolVal(sol, var) for var in model.getVars()}

        runtime = model.getSolvingTime()
    finally:
        if delete_file:
            os.remove(tmpfile_name)

    return status, solution, runtime