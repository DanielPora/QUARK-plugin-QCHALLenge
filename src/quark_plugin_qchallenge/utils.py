from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import itertools


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


class ProblemGenerator:
    """Generator for QChallenge YAML configuration files with grid-based benchmarking"""
    
    def __init__(self):
        self.problems = []
        self.qubo_problems = []
        self.mapped_problems = []
    
    def _add_configs(self, problem_name: str, param_grid: Dict[str, List[Any]], is_qubo: bool = False):
        """
        Internal method to add problem configurations with all combinations of parameters
        
        Args:
            problem_name: Name of the problem
            param_grid: Dict with parameter names and lists of values
            is_qubo: Whether this is a QUBO problem
        """
        if not param_grid:
            # Problem without parameters
            config = {problem_name: {}}
            if is_qubo:
                self.qubo_problems.append(config)
            else:
                self.problems.append(config)
            return
        
        # Filter out None values
        param_grid = {k: v for k, v in param_grid.items() if v is not None}
        
        if not param_grid:
            config = {problem_name: {}}
            if is_qubo:
                self.qubo_problems.append(config)
            else:
                self.problems.append(config)
            return
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Create cartesian product of all parameter values
        for combination in itertools.product(*param_values):
            config = {
                problem_name: dict(zip(param_names, combination))
            }
            
            if is_qubo:
                self.qubo_problems.append(config)
            else:
                self.problems.append(config)
    
    # Standard Problems
    
    def add_sensor_positioning(self, 
                              real_world_problem: bool = False,
                              # Parameters for real_world_problem = False
                              num_cols: Optional[List[int]] = None,
                              version: Optional[List[int]] = None,
                              max_radius: Optional[List[float]] = None,
                              hor_basic_distance: Optional[List[float]] = None,
                              vert_basic_dist: Optional[List[float]] = None,
                              seed: Optional[List[int]] = None,
                              # Parameters for real_world_problem = True
                              lidar_density: Optional[List[float]] = None,
                              street_point_density: Optional[List[float]] = None):
        """
        Add sensor positioning problem configurations
        
        Examples:
            # Synthetic problem
            pg.add_sensor_positioning(
                real_world_problem=False,
                num_cols=[5, 10, 15],
                version=[3],
                max_radius=[2.5],
                hor_basic_distance=[1],
                vert_basic_dist=[2],
                seed=[1, 2]
            )
            
            # Real world problem
            pg.add_sensor_positioning(
                real_world_problem=True,
                lidar_density=[0.1, 0.2],
                street_point_density=[0.1]
            )
        """
        if real_world_problem:
            # Real world problem: only lidar_density and street_point_density
            param_grid = {
                "real_world_problem": [True],
                "lidar_density": lidar_density,
                "street_point_density": street_point_density
            }
        else:
            # Synthetic problem: num_cols, version, max_radius, hor_basic_distance, vert_basic_dist, seed
            param_grid = {
                "real_world_problem": [False],
                "num_cols": num_cols,
                "version": version,
                "max_radius": max_radius,
                "hor_basic_distance": hor_basic_distance,
                "vert_basic_dist": vert_basic_dist,
                "seed": seed
            }
        
        self._add_configs("qch_sensor_positioning", param_grid, is_qubo=False)
    
    def add_auto_carrier_loading(self,
                                num_cars: Optional[List[int]] = None,
                                num_trucks: Optional[List[int]] = None,
                                seed: Optional[List[int]] = None):
        """
        Add auto carrier loading problem configurations
        
        Example:
            pg.add_auto_carrier_loading(
                num_cars=[10, 15, 20],
                num_trucks=[1, 2],
                seed=[1, 2, 3]
            )
        """
        param_grid = {
            "num_cars": num_cars,
            "num_trucks": num_trucks,
            "seed": seed
        }
        self._add_configs("qch_auto_carrier_loading", param_grid, is_qubo=False)
    
    def add_production_assignment(self,
                                 machines: Optional[List[int]] = None,
                                 jobs: Optional[List[int]] = None,
                                 seed: Optional[List[int]] = None):
        """
        Add production assignment problem configurations
        
        Example:
            pg.add_production_assignment(
                machines=[2, 3],
                jobs=[4, 6, 8],
                seed=[1, 2]
            )
        """
        param_grid = {
            "machines": machines,
            "jobs": jobs,
            "seed": seed
        }
        self._add_configs("qch_production_assignment", param_grid, is_qubo=False)
    
    def add_train_routing(self,
                         stations: Optional[List[int]] = None,
                         trains: Optional[List[int]] = None,
                         seed: Optional[List[int]] = None):
        """
        Add train routing problem configurations
        
        Example:
            pg.add_train_routing(
                stations=[2, 3, 4],
                trains=[2, 3],
                seed=[1, 2, 3]
            )
        """
        param_grid = {
            "stations": stations,
            "trains": trains,
            "seed": seed
        }
        self._add_configs("qch_train_routing", param_grid, is_qubo=False)
    
    def add_truck_loading(self,
                         num_boxes: Optional[List[int]] = None,
                         seed: Optional[List[int]] = None):
        """
        Add truck loading problem configurations
        
        Example:
            pg.add_truck_loading(
                num_boxes=[2, 5, 10],
                seed=[1, 2, 3]
            )
        """
        param_grid = {
            "num_boxes": num_boxes,
            "seed": seed
        }
        self._add_configs("qch_truck_loading", param_grid, is_qubo=False)
    
    def add_modular_production_logistics(self):
        """
        Add modular production logistics problem (no parameters)
        
        Example:
            pg.add_modular_production_logistics()
        """
        self._add_configs("qch_modular_production_logistics", {}, is_qubo=False)
    
    # QUBO Problems
    
    def add_sensor_positioning_qubo(self,
                                   real_world_problem: bool = False,
                                   # Parameters for real_world_problem = False
                                   num_cols: Optional[List[int]] = None,
                                   version: Optional[List[int]] = None,
                                   max_radius: Optional[List[float]] = None,
                                   hor_basic_distance: Optional[List[float]] = None,
                                   vert_basic_dist: Optional[List[float]] = None,
                                   # Parameters for real_world_problem = True
                                   lidar_density: Optional[List[float]] = None,
                                   street_point_density: Optional[List[float]] = None):
        """
        Add sensor positioning QUBO problem configurations
        
        Examples:
            # Synthetic problem
            pg.add_sensor_positioning_qubo(
                real_world_problem=False,
                num_cols=[5, 10],
                version=[3],
                max_radius=[2.5],
                hor_basic_distance=[1],
                vert_basic_dist=[2]
            )
            
            # Real world problem
            pg.add_sensor_positioning_qubo(
                real_world_problem=True,
                lidar_density=[0.1, 0.2],
                street_point_density=[0.1]
            )
        """
        if real_world_problem:
            # Real world problem: only lidar_density and street_point_density
            param_grid = {
                "real_world_problem": [True],
                "lidar_density": lidar_density,
                "street_point_density": street_point_density
            }
        else:
            # Synthetic problem: num_cols, version, max_radius, hor_basic_distance, vert_basic_dist
            # Note: QUBO version doesn't have seed parameter
            param_grid = {
                "real_world_problem": [False],
                "num_cols": num_cols,
                "version": version,
                "max_radius": max_radius,
                "hor_basic_distance": hor_basic_distance,
                "vert_basic_dist": vert_basic_dist
            }
        
        self._add_configs("qch_sensor_positioning_qubo", param_grid, is_qubo=True)
    
    def add_production_assignment_qubo(self,
                                      machines: Optional[List[int]] = None,
                                      jobs: Optional[List[int]] = None,
                                      seed: Optional[List[int]] = None):
        """
        Add production assignment QUBO problem configurations
        
        Example:
            pg.add_production_assignment_qubo(
                machines=[2],
                jobs=[4],
                seed=[1, 2, 3]
            )
        """
        param_grid = {
            "machines": machines,
            "jobs": jobs,
            "seed": seed
        }
        self._add_configs("qch_production_assignment_qubo", param_grid, is_qubo=True)


    def _add_mapped_configs(self, problem_name: str, param_grid: Dict[str, List[Any]]):
        if not param_grid:
            self.mapped_problems.append({problem_name: {}})
            return

        param_grid = {k: v for k, v in param_grid.items() if v is not None}
        if not param_grid:
            self.mapped_problems.append({problem_name: {}})
            return

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for combination in itertools.product(*param_values):
            config = {problem_name: dict(zip(param_names, combination))}
            self.mapped_problems.append(config)

    def add_sensor_positioning_mapped(self, lidar_density: Optional[List[float]] = None, street_point_density: Optional[List[float]] = None):
        param_grid = {
            "real_world_problem": [True],
            "lidar_density": lidar_density,
            "street_point_density": street_point_density
        }
        self._add_mapped_configs("qch_sensor_positioning", param_grid)

    def add_auto_carrier_loading_mapped(self, num_cars: Optional[List[int]] = None, num_trucks: Optional[List[int]] = None, seed: Optional[List[int]] = None):
        param_grid = {
            "num_cars": num_cars,
            "num_trucks": num_trucks,
            "seed": seed
        }
        self._add_mapped_configs("qch_auto_carrier_loading", param_grid)

    def add_production_assignment_mapped(self, machines: Optional[List[int]] = None, jobs: Optional[List[int]] = None, seed: Optional[List[int]] = None):
        param_grid = {
            "machines": machines,
            "jobs": jobs,
            "seed": seed
        }
        self._add_mapped_configs("qch_production_assignment", param_grid)

    def add_train_routing_mapped(self, stations: Optional[List[int]] = None, trains: Optional[List[int]] = None, seed: Optional[List[int]] = None):
        param_grid = {
            "stations": stations,
            "trains": trains,
            "seed": seed
        }
        self._add_mapped_configs("qch_train_routing", param_grid)
    
    # Output generation
    
    def get_stats(self) -> str:
        """Returns statistics about the generated configurations"""
        return f"Generated {len(self.problems)} standard problem configs and {len(self.qubo_problems)} QUBO problem configs"
    
    def generate_yaml(self, output_file: str = "qchallenge_config.yaml"):
        with open(output_file, 'w') as f:
            f.write('plugins: ["quark_plugin_qchallenge", "quark_plugin_luna"]\n\n')

            f.write('qchallenge_problems: &qchallenge_problems [\n')
            for problem in self.problems:
                for name, params in problem.items():
                    if params:
                        params_str = ', '.join(f'{k}: {v}' for k, v in params.items())
                        f.write(f'  "{name}": {{{params_str}}},\n')
                    else:
                        f.write(f'  "{name}": {{}},\n')
            f.write(']\n\n')

            f.write('qchallenge_QUBO_problems: &qchallenge_QUBO_problems [\n')
            for problem in self.qubo_problems:
                for name, params in problem.items():
                    if params:
                        params_str = ', '.join(f'{k}: {v}' for k, v in params.items())
                        f.write(f'  "{name}": {{{params_str}}},\n')
                    else:
                        f.write(f'  "{name}": {{}},\n')
            f.write(']\n\n')

            f.write('qchallenge_problems_mapped: &qchallenge_problems_mapped [\n')
            for problem in self.mapped_problems:
                for name, params in problem.items():
                    if params:
                        params_str = ', '.join(f'{k}: {v}' for k, v in params.items())
                        f.write(f'  "{name}": {{{params_str}}},\n')
                    else:
                        f.write(f'  "{name}": {{}},\n')
            f.write(']\n\n')

            f.write('qch_qubo_map: &qch_qubo_map "qch_qubo_map"\n\n')

            f.write('luna_classical_qubo: &luna_classical_qubo [\n')
            f.write('  "luna_sa": {num_reads: 1000},\n')
            f.write(']\n\n')

            f.write('luna_classical_lp: &luna_classical_lp [\n')
            f.write('  "luna_scip"\n')
            f.write(']\n\n')

            f.write('# Pipeline definitions\n\n')
            f.write('pipeline_qchallenge_direct: &pipeline_qchallenge_direct [*qchallenge_problems, *luna_classical_lp]\n\n')
            f.write('pipeline_qchallenge_qubo: &pipeline_qchallenge_qubo [*qchallenge_QUBO_problems, *luna_classical_qubo]\n\n')
            f.write('pipeline_qchallenge_mapping_qubo: &pipeline_qchallenge_mapping_qubo [*qchallenge_problems_mapped, *qch_qubo_map, *luna_classical_qubo]\n\n')

            f.write('pipelines: [\n')
            f.write('  *pipeline_qchallenge_direct,\n')
            f.write('  # *pipeline_qchallenge_qubo,\n')
            f.write('  # *pipeline_qchallenge_mapping_qubo,\n')
            f.write(']\n')

        print(f"✓ Configuration saved to '{output_file}'")
        print(f"  {len(self.problems)} standard, {len(self.qubo_problems)} QUBO, {len(self.mapped_problems)} mapped problems")