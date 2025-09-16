# utils/logging.py
import logging
import sys
from pathlib import Path
from loguru import logger as _logger

# Pull level from settings if available
try:
    from config.settings import settings
    _LEVEL = getattr(settings, "LOG_LEVEL", "INFO")
except Exception:
    _LEVEL = "INFO"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


class InterceptHandler(logging.Handler):
    """Route stdlib logging records into Loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        # Walk back to the original caller so file:line are accurate
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = _LEVEL) -> None:
    """Configure Loguru + intercept stdlib logging."""
    # Remove default Loguru sink
    _logger.remove()

    # Console sink
    _logger.add(
        sys.stdout,
        level=level,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )

    # File sink (rotating)
    _logger.add(
        str(LOG_FILE),
        rotation="10 MB",
        retention="30 days",
        enqueue=True,            # safer with threads/processes
        level=level,
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )

    # Intercept stdlib logging → Loguru
    logging.basicConfig(
        handlers=[InterceptHandler()],
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )

    # Quiet noisy dependencies (optional)
    for noisy in ("chromadb", "urllib3", "httpx", "openai", "langchain", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# Initialize on import so other modules can just `from utils.logging import logger`
setup_logging()

# Re-export a convenient name
logger = _logger
