"""Dedicated ThreadPoolExecutor manager for offloading blocking tasks."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar
from app.core.config import get_settings

T = TypeVar("T")


class ThreadPoolManager:
    """
    Manages the lifecycle and execution of worker threads.
    Decouples CPU-bound byte operations and blocking network calls
    from the FastAPI async event loop.
    """

    _instance: Optional["ThreadPoolManager"] = None
    _executor: Optional[ThreadPoolExecutor] = None

    def __new__(cls) -> "ThreadPoolManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, max_workers: Optional[int] = None) -> None:
        """Initialize the thread pool executor if not already running."""
        if self._executor is None:
            settings = get_settings()
            workers = max_workers or settings.MAX_THREAD_WORKERS
            self._executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="vetia-worker",
            )

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the executor, waiting for remaining tasks."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Retrieve the active ThreadPoolExecutor instance."""
        if self._executor is None:
            self.initialize()
        return self._executor  # type: ignore

    async def run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a blocking callable in the managed thread pool without blocking the async loop.
        """
        loop = asyncio.get_running_loop()
        if kwargs:
            # functools.partial to support keyword arguments in loop.run_in_executor
            from functools import partial
            bound_func = partial(func, *args, **kwargs)
            return await loop.run_in_executor(self.executor, bound_func)
        return await loop.run_in_executor(self.executor, func, *args)


_thread_pool_manager = ThreadPoolManager()


def get_thread_pool_manager() -> ThreadPoolManager:
    """Dependency provider for ThreadPoolManager."""
    return _thread_pool_manager
