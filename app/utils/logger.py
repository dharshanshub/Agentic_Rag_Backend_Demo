import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with a consistent format.

    Log level is set to DEBUG in 'dev' environment, INFO otherwise.
    Uses stdout so logs are captured cleanly by containers and cloud runtimes.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        env = os.getenv("ENVIRONMENT", "dev").lower()
        logger.setLevel(logging.DEBUG if env == "dev" else logging.INFO)

    return logger
