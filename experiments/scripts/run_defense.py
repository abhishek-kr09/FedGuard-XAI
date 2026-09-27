"""Run robust client-update aggregation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.server import aggregate_updates

if __name__ == "__main__":
    updates = [[0.2, 0.4], [0.3, 0.5], [0.25, 0.45], [9.0, 9.0], [0.22, 0.43]]
    print({"experiment": "defense", "aggregated_update": aggregate_updates(updates, "trimmed_mean", 0.2).tolist()})
