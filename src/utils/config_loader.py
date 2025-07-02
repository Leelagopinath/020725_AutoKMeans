# File: src/utils/config_loader.py

import yaml
import os

def load_config(file_path="config/default.yaml"):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file doesn't exist
        return {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "clustering": {
                "default_metric": "Euclidean",
                "max_iterations": 100,
                "metric_params": {
                    "Mahalanobis": {"use_pseudo_inverse": True},
                    "Gower": {"categorical_features": []},
                    "K-Prototypes": {"gamma": 0.5}
                }
            }
        }