#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL="postgresql://postgres_user:secure_pass@localhost:5432/sciforge_db"

# Find the absolute canonical path of the folder containing this shell script
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CRITICAL FIX: Direct the Python Path to search inside the src/ directory
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

echo "=== 3. Code Quality & Code Formatting Validation ==="
ruff check . --fix || echo "Ruff flagged warnings, continuing automation loop..."
ruff format .

echo "=== 4. Seeding Database Records ==="
python "${REPO_ROOT}/apps/run_materials_pipeline.py" --material-ids mp-100 mp-200 mp-300

echo "=== 5. Final SQL Schema Extraction Check ==="
PGPASSWORD="secure_pass" psql -h localhost -U postgres_user -d sciforge_db -c "SELECT * FROM materials;"
