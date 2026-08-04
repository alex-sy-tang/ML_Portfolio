from pathlib import Path

# Single source of truth for project paths, imported by functions.py, app.py,
# the notebook, and scripts/run_weekly_pipeline.py (once it exists) instead of
# each redefining PROJECT_ROOT/DATA_DIR separately.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
