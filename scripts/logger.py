"""
Structured logging utility for Vedic Shloka Intelligence System
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_dir: Path = None, level: str = "INFO") -> logging.Logger:
    """
    Set up a structured logger with both console and file handlers.

    Args:
        name: Logger name (typically module name)
        log_dir: Directory to write log files (optional)
        level: Logging level string

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    # Windows consoles default to a legacy code page (e.g. cp1252) that cannot
    # encode Devanagari or symbols like '→', making every such log call emit a
    # UnicodeEncodeError traceback. Reconfigure stdout to UTF-8 if possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class JSONLogger:
    """
    Writes structured JSON log entries to a JSONL file for machine-readable audit trails.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **kwargs):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            **kwargs,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
