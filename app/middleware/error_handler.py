from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import get_request_id
from app.core.exceptions import ServiceException
from loguru import logger


async def service_exception_handler(
    request: Request, exc: ServiceException
) -> JSONResponse:
    logger.error(
        f"Business logic error: {exc.detail} | Exception: {exc.__class__.__name__}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": get_request_id(),
            "error_type": exc.__class__.__name__,
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.error(f"Input validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "request_id": get_request_id(),
            "error_type": "ValidationError",
        },
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    logger.error(f"HTTP exception: {exc.detail} | Status: {exc.status_code}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": get_request_id(),
            "error_type": "HTTPException",
        },
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.opt(exception=exc).error("Unhandled internal server error occurred.")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal server error occurred.",
            "request_id": get_request_id(),
            "error_type": exc.__class__.__name__,
        },
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Registers the exception handlers onto the FastAPI application.
    """
    app.add_exception_handler(ServiceException, service_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
