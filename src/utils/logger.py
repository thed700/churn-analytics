"""
Structured logging configuration using loguru.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_file: str = "logs/churn_pipeline.log", level: str = "INFO") -> None:
    """Configure loguru logger with console and file handlers."""
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        level=level,
    )

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        level=level,
    )

    return logger
