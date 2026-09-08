"""
logger.py

Centralized logging configuration for CampusGuide AI.
"""

import logging
import sys


def setup_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name (str): Usually pass __name__

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger