import yaml
import itertools
from typing import Dict, List, Any

class QChallengeConfigGenerator:
    """Generator für QChallenge YAML-Konfigurationsdateien mit Benchmarking-Support"""
    
    AVAILABLE_PROBLEMS = {
        "qch_sensor_positioning": {
            "params": ["num_cols", "version", "max_radius", "hor_basic_distance", "vert_basic_dist"],
            "description": "Sensor Positioning Problem"
        },
        "qch_sensor_positioning_real": {
            "params": ["real_world_problem", "lidar_density", "street_point_density"],
            "description": "Sensor Positioning (Real World)"
        },
        "qch_auto_carrier_loading": {
            "params": ["num_cars", "num_trucks", "seed"],
            "description": "Auto Carrier Loading"
        },
        "qch_production_assignment": {
            "params": ["machines", "jobs", "seed"],
            "description": "Production Assignment"
        },
        "qch_train_routing": {
            "params": ["stations", "trains", "seed"],
            "description": "Train Routing"
        },
        "qch_truck_loading": {
            "params": ["num_boxes", "seed"],
            "description": "Truck Loading"
        },
        "qch_modular_production_logistics": {
            "params": [],
            "description": "Modular Production Logistics"
        }
    }
    
    AVAILABLE_QUBO_PROBLEMS = {
        "qch_sensor_positioning_qubo": {
            "params": ["num_cols", "version", "max_radius", "hor_basic_distance", "vert_basic_dist"],
            "description": "Sensor Positioning QUBO"
        },
        "qch_sensor_positioning_qubo_real": {
            "params": ["real_world_problem", "lidar_density", "street_point_density"],
            "description": "Sensor Positioning QUBO (Real World)"
        },
        "qch_production_assignment_qubo": {
            "params": ["machines", "jobs", "seed"],
            "description": "Production Assignment QUBO"
        }
    }
    
    def __init__(self):
        self.problems = []
        self.qubo_problems = []
    
    def print_available_problems(self):
        """Zeigt alle verfügbaren Problem-Typen an"""
        print("\n=== Verfügbare Standard-Probleme ===")
        for i, (name, info) in enumerate(self.AVAILABLE_PROBLEMS.items(), 1):
            print(f"{i}. {name}")
            print(f"   {info['description']}")
            if info['params']:
                print(f"   Parameter: {', '.join(info['params'])}")
            print()
        
        print("\n=== Verfügbare QUBO-Probleme ===")
        for i, (name, info) in enumerate(self.AVAILABLE_QUBO_PROBLEMS.items(), 1):
            print(f"{i}. {name}")
            print(f"   {info['description']}")
            if info['params']:
                print(f"   Parameter: {', '.join(info['params'])}")
            print()
    
    def add_problem_configs(self, problem_name: str, param_lists: Dict[str, List[Any]], 
                           is_qubo: bool = False):
        """
        Fügt Problem-Konfigurationen mit allen Kombinationen der Parameter hinzu
        
        Args:
            problem_name: Name des Problems (z.B. "qch_sensor_positioning")
            param_lists: Dict mit Parameter-Namen und Listen von Werten
            is_qubo: Ob es sich um ein QUBO-Problem handelt
        """
        # Spezialbehandlung für real_world_problem
        if "real_world_problem" in param_lists:
            problem_name = problem_name.replace("_qubo", "") + ("_qubo" if is_qubo else "")
        
        # Generiere alle Kombinationen
        if not param_lists:
            # Problem ohne Parameter
            config = {problem_name: {}}
            if is_qubo:
                self.qubo_problems.append(config)
            else:
                self.problems.append(config)
            return
        
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        
        # Erstelle kartesisches Produkt aller Parameterwerte
        for combination in itertools.product(*param_values):
            config = {
                problem_name: dict(zip(param_names, combination))
            }
            
            if is_qubo:
                self.qubo_problems.append(config)
            else:
                self.problems.append(config)
    
    def interactive_mode(self):
        """Interaktiver Modus zum Erstellen der Konfiguration"""
        print("=== QChallenge Config Generator - Benchmarking Mode ===\n")
        
        # Standard-Probleme
        print("Möchtest du Standard-Probleme hinzufügen? (j/n): ", end="")
        if input().lower() == 'j':
            self._add_problems_interactive(is_qubo=False)
        
        # QUBO-Probleme
        print("\nMöchtest du QUBO-Probleme hinzufügen? (j/n): ", end="")
        if input().lower() == 'j':
            self._add_problems_interactive(is_qubo=True)
    
    def _add_problems_interactive(self, is_qubo: bool):
        """Hilfsfunktion für interaktives Hinzufügen von Problemen"""
        problems_dict = self.AVAILABLE_QUBO_PROBLEMS if is_qubo else self.AVAILABLE_PROBLEMS
        problem_type = "QUBO" if is_qubo else "Standard"
        
        while True:
            print(f"\n=== {problem_type}-Problem hinzufügen ===")
            print("Verfügbare Probleme:")
            problem_list = list(problems_dict.keys())
            for i, name in enumerate(problem_list, 1):
                print(f"{i}. {name} - {problems_dict[name]['description']}")
            
            print("\nWähle Problem-Nummer (oder 0 zum Beenden): ", end="")
            choice = int(input())
            
            if choice == 0:
                break
            
            if 1 <= choice <= len(problem_list):
                problem_name = problem_list[choice - 1]
                info = problems_dict[problem_name]
                
                print(f"\n--- {problem_name} ---")
                
                if not info['params']:
                    print("Dieses Problem hat keine Parameter.")
                    self.add_problem_configs(problem_name, {}, is_qubo)
                    print("✓ Problem hinzugefügt!")
                    continue
                
                print(f"Parameter: {', '.join(info['params'])}")
                print("\nFür Benchmarking kannst du für jeden Parameter mehrere Werte angeben.")
                print("Beispiel: 1,2,3 oder 0.1,0.5,1.0")
                
                param_lists = {}
                for param in info['params']:
                    print(f"\n{param}: ", end="")
                    value_input = input().strip()
                    
                    if not value_input:
                        continue
                    
                    # Parse Werte
                    values = []
                    for v in value_input.split(','):
                        v = v.strip()
                        # Versuche verschiedene Typen
                        if v.lower() == 'true':
                            values.append(True)
                        elif v.lower() == 'false':
                            values.append(False)
                        else:
                            try:
                                # Integer
                                values.append(int(v))
                            except ValueError:
                                try:
                                    # Float
                                    values.append(float(v))
                                except ValueError:
                                    # String
                                    values.append(v)
                    
                    if values:
                        param_lists[param] = values
                
                if param_lists:
                    self.add_problem_configs(problem_name, param_lists, is_qubo)
                    num_configs = len(list(itertools.product(*param_lists.values())))
                    print(f"✓ {num_configs} Konfiguration(en) hinzugefügt!")
    
    def generate_yaml(self, output_file: str = "qchallenge_config.yaml"):
        """Generiert die finale YAML-Datei"""
        config = {
            "plugins": ["quark_plugin_qchallenge", "quark_plugin_luna"],
            "qchallenge_problems": self.problems,
            "qchallenge_QUBO_problems": self.qubo_problems,
            "qch_qubo_map": "qch_qubo_map",
            "luna_classical_qubo": [
                {"luna_sa": {"num_reads": 1000}}
            ],
            "luna_classical_lp": ["luna_scip"],
            "pipeline_qchallenge_direct": ["*qchallenge_problems", "*luna_classical_lp"],
            "pipeline_qchallenge_qubo": ["*qchallenge_QUBO_problems", "*luna_classical_qubo"],
            "pipeline_qchallenge_mapping_qubo": ["*qchallenge_problems", "*qch_qubo_map", "*luna_classical_qubo"],
            "pipelines": ["*pipeline_qchallenge_direct"]
        }
        
        # YAML mit Anchors manuell erstellen
        with open(output_file, 'w') as f:
            f.write('plugins: ["quark_plugin_qchallenge", "quark_plugin_luna"]\n\n')
            
            # Problems
            f.write('qchallenge_problems: &qchallenge_problems [\n')
            for problem in self.problems:
                for name, params in problem.items():
                    if params:
                        params_str = ', '.join(f'{k}: {v}' for k, v in params.items())
                        f.write(f'  "{name}": {{{params_str}}},\n')
                    else:
                        f.write(f'  "{name}": {{}},\n')
            f.write(']\n\n')
            
            # QUBO Problems
            f.write('qchallenge_QUBO_problems: &qchallenge_QUBO_problems [\n')
            for problem in self.qubo_problems:
                for name, params in problem.items():
                    if params:
                        params_str = ', '.join(f'{k}: {v}' for k, v in params.items())
                        f.write(f'  "{name}": {{{params_str}}},\n')
                    else:
                        f.write(f'  "{name}": {{}},\n')
            f.write(']\n\n')
            
            # Rest der Config
            f.write('qch_qubo_map: &qch_qubo_map "qch_qubo_map"\n\n')
            f.write('luna_classical_qubo: &luna_classical_qubo [\n')
            f.write('  "luna_sa": {num_reads: 1000},\n')
            f.write(']\n\n')
            f.write('luna_classical_lp: &luna_classical_lp [\n')
            f.write('  "luna_scip"\n')
            f.write(']\n\n')
            f.write('# Pipeline definitions\n\n')
            f.write('pipeline_qchallenge_direct: &pipeline_qchallenge_direct [*qchallenge_problems, *luna_classical_lp]\n')
            f.write('pipeline_qchallenge_qubo: &pipeline_qchallenge_qubo [*qchallenge_QUBO_problems, *luna_classical_qubo]\n')
            f.write('pipeline_qchallenge_mapping_qubo: &pipeline_qchallenge_mapping_qubo [*qchallenge_problems, *qch_qubo_map, *luna_classical_qubo]\n\n')
            f.write('pipelines: [\n')
            f.write('  *pipeline_qchallenge_direct,\n')
            f.write(']\n')
        
        print(f"\n✓ Konfiguration erfolgreich in '{output_file}' gespeichert!")
        print(f"  Standard-Probleme: {len(self.problems)}")
        print(f"  QUBO-Probleme: {len(self.qubo_problems)}")


def main():
    generator = QChallengeConfigGenerator()
    
    # Beispiel: Programmatische Nutzung
    print("=== Programmatischer Modus (Beispiel) ===\n")
    
    # Sensor Positioning mit verschiedenen Parametern
    generator.add_problem_configs(
        "qch_sensor_positioning",
        {
            "num_cols": [5, 10],
            "version": [3],
            "max_radius": [2.5],
            "hor_basic_distance": [1],
            "vert_basic_dist": [2]
        }
    )
    
    # Production Assignment QUBO mit verschiedenen Seeds
    generator.add_problem_configs(
        "qch_production_assignment_qubo",
        {
            "machines": [2],
            "jobs": [4],
            "seed": [1, 2, 3]
        },
        is_qubo=True
    )
    
    print(f"Generierte Konfigurationen: {len(generator.problems)} Standard, {len(generator.qubo_problems)} QUBO\n")
    
    # Interaktiver Modus
    print("\nMöchtest du weitere Probleme im interaktiven Modus hinzufügen? (j/n): ", end="")
    if input().lower() == 'j':
        generator.interactive_mode()
    
    # YAML generieren
    generator.generate_yaml()


if __name__ == "__main__":
    main()