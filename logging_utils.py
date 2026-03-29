"""
LOGGING UTILITIES — Structured logging and observability for production.
Provides JSON logging with correlation IDs and per-stage metrics.
"""
import os, sys, logging, json, time, traceback
from datetime import datetime
from functools import wraps

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED LOG FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredFormatter(logging.Formatter):
    """Outputs JSON-formatted logs for machine parsing."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add stage info if present
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage

        # Add timing info if present
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter with colors for terminal."""

    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Format timestamp
        ts = self.formatTime(record, "%H:%M:%S")

        # Add correlation ID if present
        corr = ""
        if hasattr(record, "correlation_id") and record.correlation_id:
            corr = f"[{record.correlation_id[:8]}] "

        return f"{color}{ts}{reset} {corr}{record.getMessage()}"


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class StageTracker:
    """
    Tracks pipeline execution with timing and metrics.
    Use as a context manager for automatic timing.
    """

    def __init__(self, name: str, logger: logging.Logger, correlation_id: str = None):
        self.name = name
        self.logger = logger
        self.correlation_id = correlation_id or generate_correlation_id()
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def __enter__(self):
        self.start_time = time.time()
        extra = {"stage": self.name, "correlation_id": self.correlation_id}
        self.logger.info(f"[{self.name}] Starting...", extra=extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        extra = {
            "stage": self.name,
            "correlation_id": self.correlation_id,
            "duration_ms": round(duration_ms, 2),
        }

        if exc_type:
            extra["error"] = str(exc_val)
            self.logger.error(
                f"[{self.name}] Failed after {duration_ms:.0f}ms: {exc_val}",
                extra=extra,
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            self.logger.info(
                f"[{self.name}] Completed in {duration_ms:.0f}ms",
                extra=extra
            )

        return False  # Don't suppress exceptions

    def log_metric(self, key: str, value):
        """Log a metric for this stage."""
        self.metrics[key] = value

    def get_metrics(self) -> dict:
        """Get all metrics for this stage."""
        return {
            "stage": self.name,
            "duration_ms": round((self.end_time - self.start_time) * 1000, 2) if self.end_time else None,
            "metrics": self.metrics,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUN CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineContext:
    """
    Context for a pipeline run. Provides correlation ID and stage tracking.
    Pass this through the pipeline for consistent observability.
    """

    def __init__(self, url: str, target_lang: str, source_lang: str):
        self.correlation_id = generate_correlation_id()
        self.url = url
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.stages = []
        self.start_time = time.time()

    def start_stage(self, name: str, logger: logging.Logger) -> StageTracker:
        """Start a new stage with timing."""
        tracker = StageTracker(name, logger, self.correlation_id)
        self.stages.append(tracker)
        return tracker

    def get_summary(self) -> dict:
        """Get pipeline execution summary."""
        total_duration = (time.time() - self.start_time) * 1000
        stage_summaries = []

        for s in self.stages:
            summary = s.get_metrics()
            del summary["metrics"]  # Clean up
            stage_summaries.append(summary)

        return {
            "correlation_id": self.correlation_id,
            "url": self.url,
            "target_lang": self.target_lang,
            "source_lang": self.source_lang,
            "total_duration_ms": round(total_duration, 2),
            "stages": stage_summaries,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_correlation_id() -> str:
    """Generate a unique correlation ID for a pipeline run."""
    import uuid
    return str(uuid.uuid4())


def setup_logging(structured: bool = False, level: int = logging.INFO):
    """
    Setup logging for the application.

    Args:
        structured: If True, output JSON. If False, human-readable with colors.
        level: Logging level (default INFO)
    """
    handler = logging.StreamHandler(sys.stdout)

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = HumanFormatter()

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []
    root_logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_duration(logger: logging.Logger, stage: str, correlation_id: str = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            extra = {"stage": stage}
            if correlation_id:
                extra["correlation_id"] = correlation_id

            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - t0) * 1000
                logger.info(
                    f"[{stage}] {func.__name__} completed in {duration_ms:.0f}ms",
                    extra=extra
                )
                return result
            except Exception as e:
                duration_ms = (time.time() - t0) * 1000
                logger.error(
                    f"[{stage}] {func.__name__} failed after {duration_ms:.0f}ms: {e}",
                    extra=extra
                )
                raise
        return wrapper
    return decorator
