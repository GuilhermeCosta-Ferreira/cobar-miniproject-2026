# ================================================================
# 0. Section: IMPORTS
# ================================================================
import yaml

from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
