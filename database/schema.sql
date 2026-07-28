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

-- Index to optimize quick queries by material when loading the PINN
CREATE INDEX idx_materials_formula ON materials(formula);
