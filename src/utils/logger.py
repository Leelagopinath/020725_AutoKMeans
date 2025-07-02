# File: src/utils/logger.py

import logging
import sys
from src.utils.config_loader import load_config

def get_logger(name):
    config = load_config()
    logging_config = config.get("logging", {})
    
    logger = logging.getLogger(name)
    logger.setLevel(logging_config.get("level", "INFO"))
    
    formatter = logging.Formatter(
        logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler (in production)
    if not sys.argv[0].endswith('streamlit'):
        fh = logging.FileHandler('kmeans_automation.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger