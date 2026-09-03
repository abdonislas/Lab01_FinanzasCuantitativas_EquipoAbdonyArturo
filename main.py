from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# para poder importar src/ sin importar desde donde se corra el script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import plots
from src.model import (ModelParams, expected_informed_loss_ask,
                       expected_informed_loss_bid, expected_utility,
                       optimize_quotes, optimize_quotes_grid,
                       sensitivity_to_pi_informed)
from src.simulation import (DEFAULT_SEED, expected_inventory_drift,
                            monte_carlo_stats, run_monte_carlo,
                            simulate_trades, summarize_run,
                            validate_against_theory)
