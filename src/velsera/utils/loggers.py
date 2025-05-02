import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


def log_time(func: Callable) -> Callable:
    """Decorator to log the time taken by a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Time taken for {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper
