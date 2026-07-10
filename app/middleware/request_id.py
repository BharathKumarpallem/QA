import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import request_id_var
from loguru import logger


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique request ID to each incoming request,
    stores it in context, logs request lifecycles, and returns it in headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind request_id to the async context variable
        token = request_id_var.set(request_id)

        start_time = time.perf_counter()
        logger.info(
            f"Incoming Request: {request.method} {request.url.path} {request.query_params}"
        )

        try:
            response: Response = await call_next(request)
            process_time = time.perf_counter() - start_time
            logger.info(
                f"Outgoing Response: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Latency: {process_time:.4f}s"
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            process_time = time.perf_counter() - start_time
            logger.exception(
                f"Request Failed: {request.method} {request.url.path} "
                f"Latency: {process_time:.4f}s | Error: {str(e)}"
            )
            raise e
        finally:
            # Clean up token context to prevent leaks
            request_id_var.reset(token)
