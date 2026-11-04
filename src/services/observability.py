import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.config import settings

logger = logging.getLogger(__name__)

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    class BaseCallbackHandler:  # type: ignore
        def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
            pass

        def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
            pass

        def on_chain_error(self, error: Any, **kwargs: Any) -> None:
            pass

        def on_llm_start(self, serialized: Any, prompts: Any, **kwargs: Any) -> None:
            pass

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            pass

        def on_llm_error(self, error: Any, **kwargs: Any) -> None:
            pass

        def on_retriever_start(self, serialized: Any, query: Any, **kwargs: Any) -> None:
            pass

        def on_retriever_end(self, documents: Any, **kwargs: Any) -> None:
            pass

        def on_retriever_error(self, error: Any, **kwargs: Any) -> None:
            pass


class NoOpSpan:
    """Safe no-op representation of a Langfuse span."""

    def __init__(self, name: str = "", **kwargs: Any):
        self.name = name
        self.kwargs = kwargs

    def update(self, **kwargs: Any) -> "NoOpSpan":
        return self

    def end(self, **kwargs: Any) -> "NoOpSpan":
        return self

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NoOpGeneration(NoOpSpan):
    """Safe no-op representation of a Langfuse generation."""

    pass


class NoOpTrace(NoOpSpan):
    """Safe no-op representation of a Langfuse trace."""

    def span(self, name: str = "", **kwargs: Any) -> NoOpSpan:
        return NoOpSpan(name=name, **kwargs)

    def generation(self, name: str = "", **kwargs: Any) -> NoOpGeneration:
        return NoOpGeneration(name=name, **kwargs)


class NoOpCallbackHandler(BaseCallbackHandler):
    """Safe no-op LangChain callback handler when Langfuse is not enabled."""

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_retriever_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        pass


class ObservabilityService:
    """Service managing Langfuse client, trace/span creation, and LangChain callbacks."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        client: Any | None = None,
    ):
        self.public_key = (
            public_key if public_key is not None else settings.langfuse_public_key
        )
        self.secret_key = (
            secret_key if secret_key is not None else settings.langfuse_secret_key
        )
        self.host = host if host is not None else settings.langfuse_host
        self._custom_client = client
        self._client: Any | None = None
        self._init_client()

    @property
    def is_enabled(self) -> bool:
        if self._custom_client is not None:
            return True
        return bool(self.public_key and self.secret_key)

    def _init_client(self) -> None:
        if self._custom_client is not None:
            self._client = self._custom_client
            return
        if not (self.public_key and self.secret_key):
            self._client = None
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse client: %s", exc)
            self._client = None

    def get_client(self) -> Any:
        return self._client

    def create_trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not self.is_enabled or self._client is None:
            return NoOpTrace(name=name)

        try:
            return self._client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or [],
                **kwargs,
            )
        except Exception as exc:
            logger.warning("Langfuse create_trace failed: %s", exc)
            return NoOpTrace(name=name)

    def get_langchain_callback(
        self,
        trace_name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseCallbackHandler:
        if not self.is_enabled:
            return NoOpCallbackHandler()

        try:
            from langfuse.callback import CallbackHandler

            return CallbackHandler(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                trace_name=trace_name,
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Langfuse CallbackHandler initialization failed: %s", exc)
            return NoOpCallbackHandler()

    def flush(self) -> None:
        if self.is_enabled and self._client is not None and hasattr(self._client, "flush"):
            try:
                self._client.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)


_observability_service: ObservabilityService | None = None


def get_observability_service() -> ObservabilityService:
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware to trace incoming HTTP requests via Langfuse."""

    def __init__(
        self,
        app: Any,
        observability_service: ObservabilityService | None = None,
    ):
        super().__init__(app)
        self.observability_service = observability_service or get_observability_service()

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        user_id = (
            request.headers.get("x-user-id")
            or request.headers.get("user-id")
            or request.query_params.get("user_id")
        )
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("session-id")
            or request.query_params.get("session_id")
        )
        tenant_id = (
            request.headers.get("x-tenant-id")
            or request.headers.get("tenant-id")
            or request.headers.get("x-namespace")
            or request.query_params.get("tenant_id")
            or request.query_params.get("namespace")
        )

        method = request.method
        path = request.url.path
        trace_name = f"{method} {path}"

        tags = ["http", method]
        metadata: dict[str, Any] = {
            "method": method,
            "path": path,
            "query_params": dict(request.query_params),
        }
        if tenant_id:
            metadata["tenant_id"] = tenant_id
            tags.append(f"tenant:{tenant_id}")

        trace = self.observability_service.create_trace(
            name=trace_name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
        )
        request.state.trace = trace

        start_time = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            status_code = 500
            if hasattr(trace, "update"):
                trace.update(
                    level="ERROR",
                    status_message=str(exc),
                )
            raise
        finally:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            metadata["status_code"] = status_code
            metadata["latency_ms"] = latency_ms
            tags.append(str(status_code))
            if hasattr(trace, "update"):
                trace.update(
                    metadata=metadata,
                    tags=tags,
                )
            if hasattr(trace, "end"):
                trace.end()
