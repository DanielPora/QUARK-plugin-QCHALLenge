from utils import ProblemGenerator

if __name__ == "__main__":
    pg = ProblemGenerator()
    
    # ----------------------------
    # Standard Problems
    # ----------------------------
    pg.add_sensor_positioning(
        real_world_problem=False,
        num_cols=[5, 10],
        version=[2, 3],
        max_radius=[2.5],
        hor_basic_distance=[1],
        vert_basic_dist=[2],
        seed=[1, 11]
    )

    pg.add_sensor_positioning(
        real_world_problem=True,
        lidar_density=[0.1, 0.2],
        street_point_density=[0.1, 0.15]
    )

    pg.add_truck_loading(
        num_boxes=[2, 5, 10],
        seed=[1, 2]
    )
    
    pg.add_auto_carrier_loading(
        num_cars=[10],
        num_trucks=[1],
        seed=[1]
    )
    
    pg.add_modular_production_logistics()

    # ----------------------------
    # QUBO Problems
    # ----------------------------
    pg.add_sensor_positioning_qubo(
        real_world_problem=False,
        num_cols=[5, 10],
        version=[3],
        max_radius=[2.5],
        hor_basic_distance=[1],
        vert_basic_dist=[2]
    )
    
    pg.add_sensor_positioning_qubo(
        real_world_problem=True,
        lidar_density=[0.1],
        street_point_density=[0.1]
    )
    
    pg.add_production_assignment_qubo(
        machines=[2, 3],
        jobs=[4, 6],
        seed=[1, 11, 345]
    )

    # ----------------------------
    # Mapped Problems
    # ----------------------------
    pg.add_sensor_positioning_mapped(
        lidar_density=[0.1, 0.2],
        street_point_density=[0.1, 0.15]
    )
    
    pg.add_auto_carrier_loading_mapped(
        num_cars=[10],
        num_trucks=[1],
        seed=[1]
    )
    
    pg.add_production_assignment_mapped(
        machines=[2, 3],
        jobs=[4, 6],
        seed=[1, 11, 345]
    )
    
    pg.add_train_routing_mapped(
        stations=[2, 4],
        trains=[2],
        seed=[1]
    )

    # ----------------------------
    # Generate YAML
    # ----------------------------
    print(pg.get_stats())
    pg.generate_yaml("benchmark_config.yaml")
