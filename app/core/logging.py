import logging
import sys
from contextvars import ContextVar
from loguru import logger

# ContextVar to store request ID for the current async task
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return request_id_var.get()


class InterceptHandler(logging.Handler):
    """
    Intercepts standard library logging calls and routes them to loguru.
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging():
    # Remove existing handlers from standard root logger
    logging.root.handlers = []

    # Intercept all logs from standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

    # Intercept uvicorn logs specifically and prevent duplicate logging
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    # Configure Loguru format and handlers
    # Include Request ID in format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "ReqID: <magenta>{extra[request_id]}</magenta> | "
        "<level>{message}</level>"
    )

    # Re-initialize loguru
    logger.remove()
    logger.add(
        sys.stdout,
        format=log_format,
        level="INFO",
        enqueue=True,
    )

    # Configure the patcher to dynamically inject request ID from context variable
    logger.configure(
        patcher=lambda record: record["extra"].update(request_id=get_request_id())
    )
