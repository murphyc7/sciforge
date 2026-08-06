-- Core table for material properties fetched by the ETL pipeline
CREATE TABLE materials (
    material_id VARCHAR(50) PRIMARY KEY,
    formula VARCHAR(20) NOT NULL,
    space_group INT NOT NULL,
    effective_mass_me DOUBLE PRECISION NOT NULL,
    permittivity DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to log PINN simulation training metadata and loss history
CREATE TABLE pinn_simulation_logs (
    simulation_id SERIAL PRIMARY KEY,
    material_id VARCHAR(50) REFERENCES materials(material_id) ON DELETE CASCADE,
    epochs_trained INT NOT NULL,
    final_physics_loss DOUBLE PRECISION NOT NULL,
    final_data_loss DOUBLE PRECISION NOT NULL,
    model_weights_path VARCHAR(255) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index to optimise quick queries by material when loading the PINN
CREATE INDEX idx_materials_formula ON materials(formula);

-- Table to manage training iterations, hyperparameters, and model paths
CREATE TABLE carrier_pinn_simulation_registry (
    run_id SERIAL PRIMARY KEY,
    material_id VARCHAR(50) NOT NULL REFERENCES materials(material_id) ON DELETE CASCADE,
    engine_mode VARCHAR(20) NOT NULL,          -- 'constant' or 'coupled'
    epochs_trained INT NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    final_total_loss DOUBLE PRECISION NOT NULL,
    final_bc_loss DOUBLE PRECISION NOT NULL,
    final_physics_loss DOUBLE PRECISION NOT NULL,
    execution_time_seconds DOUBLE PRECISION NOT NULL,
    model_artifact_path VARCHAR(512) NOT NULL, -- Path to versioned ONNX/Pt binary file
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optimise indexed searches for comparing simulation runs by material or engine types
CREATE INDEX idx_sim_registry_material ON carrier_pinn_simulation_registry(material_id);
CREATE INDEX idx_sim_registry_engine ON carrier_pinn_simulation_registry(engine_mode);
