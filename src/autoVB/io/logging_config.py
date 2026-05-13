from __future__ import annotations

import logging
import warnings
from typing import TextIO


DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"


def configure_logging(
    level: int = logging.INFO,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure package-level logging and warning forwarding."""
    root_logger = logging.getLogger()
    if force or not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format=DEFAULT_LOG_FORMAT,
            stream=stream,
            force=force,
        )
    else:
        root_logger.setLevel(level)

    logging.captureWarnings(True)
    warnings.simplefilter("default")


def get_logger(name: str) -> logging.Logger:
    """Return a logger using the package logging configuration."""
    return logging.getLogger(name)


def warn_user(
    message: str,
    category: type[Warning] = RuntimeWarning,
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a user-visible warning without printing directly."""
    warnings.warn(message, category, stacklevel=stacklevel)

