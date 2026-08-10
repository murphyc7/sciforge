#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL="postgresql://postgres_user:secure_pass@localhost:5432/sciforge_db"

# Find the absolute canonical path of the folder containing this shell script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Direct the Python Path to search inside the src/ directory
export PYTHONPATH="${REPO_ROOT}/src"

echo "=== System Path Resolution ==="
echo "Repository Root: ${REPO_ROOT}"
echo "Injected PYTHONPATH: ${PYTHONPATH}"

echo "=== 1 & 2. Verifying Native Database Readiness ==="
if ! pg_isready -h localhost -p 5432 -U postgres_user -d sciforge_db &>/dev/null; then
    echo "Error: Local PostgreSQL service is not running or credentials do not match."
    exit 1
fi
echo "Native Database is ready!"

# Ensure our local system artifact folders exist cleanly on disk
mkdir -p "${REPO_ROOT}/artifacts/models"

echo "Applying database registry schema updates..."
PGPASSWORD="secure_pass" psql -h localhost -U postgres_user -d sciforge_db -c "
CREATE TABLE IF NOT EXISTS carrier_pinn_simulation_registry (
    run_id SERIAL PRIMARY KEY,
    material_id VARCHAR(50) NOT NULL REFERENCES materials(material_id) ON DELETE CASCADE,
    engine_mode VARCHAR(20) NOT NULL,
    epochs_trained INT NOT NULL,
    learning_rate DOUBLE PRECISION NOT NULL,
    final_total_loss DOUBLE PRECISION NOT NULL,
    final_bc_loss DOUBLE PRECISION NOT NULL,
    final_physics_loss DOUBLE PRECISION NOT NULL,
    execution_time_seconds DOUBLE PRECISION NOT NULL,
    model_artifact_path VARCHAR(512) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);"

# Force truncate the tracking registry table to clear any lingering, uncommitted thread locks
echo "Truncating stale ledger configurations..."
PGPASSWORD="secure_pass" psql -h localhost -U postgres_user -d sciforge_db -c "TRUNCATE TABLE carrier_pinn_simulation_registry RESTART IDENTITY CASCADE;"

echo "=== 3. Code Quality & Code Formatting Validation ==="
ruff check . --fix || echo "Ruff flagged warnings, continuing automation loop..."
ruff format .

echo "Wiping artifacts..."
rm -f "${REPO_ROOT}/artifacts/models"/*

echo "=== 4. Seeding Database Records ==="
python "${REPO_ROOT}/apps/run_materials_pipeline.py" --material-ids mp-100

echo "=== 5. Final SQL Schema Extraction Check ==="
PGPASSWORD="secure_pass" psql -h localhost -U postgres_user -d sciforge_db -c "SELECT * FROM materials;"

echo "=== 6a. Executing Constant Field PINN Simulation ==="
python "${REPO_ROOT}/apps/run_carrier_simulation.py" --material-id mp-100 --engine constant --epochs 500

echo "=== 6b. Executing Self-Consistent Poisson-Coupled PINN Simulation ==="
python "${REPO_ROOT}/apps/run_carrier_simulation.py" --material-id mp-100 --engine coupled --epochs 500

echo "=== 7. Final SQL Simulation Ledger Registry Check ==="
PGPASSWORD="secure_pass" psql -h localhost -U postgres_user -d sciforge_db -x -c "SELECT run_id, material_id, engine_mode, final_total_loss, execution_time_seconds, model_artifact_path FROM carrier_pinn_simulation_registry;"
