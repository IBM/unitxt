import functools
import hashlib
import json
import os

from .logging_utils import get_logger

logger = get_logger()

# Check if caching is enabled via environment variable
CACHE_LOCATION = os.getenv("UNITXT_CACHE_LOCATION")

# Set max cache size to 10GB or the value of env var MAX_CACHE_SIZE
MAX_CACHE_SIZE = os.getenv("MAX_CACHE_SIZE", 10 * 1024**3)


_cache_instance = None


def get_cache():
    """Returns a singleton cache instance, initializing it if necessary."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = Cache()
    return _cache_instance


def generate_cache_key(*args, **kwargs):
    """Generate a stable hashable cache key for various input types.

    :param args: Positional arguments of the function.
    :param kwargs: Keyword arguments of the function.
    :return: A hashed key as a string.
    """
    try:
        # Convert args and kwargs to a JSON string (sorted to ensure consistency)
        serialized = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
    except TypeError:
        # Fallback for non-serializable objects
        serialized = repr((args, kwargs))

    # Hash the serialized data
    return hashlib.md5(serialized.encode()).hexdigest()


class Cache:
    """A class that provides disk-based caching functionality for a given function."""

    def __init__(self):
        """Initializes the cache.

        If `CACHE_LOCATION` (os.getenv("UNITXT_CACHE_LOCATION") is set, a disk-based
        cache is created using `diskcache`.

        Args:
            None

        Returns:
            None
        """
        if CACHE_LOCATION:
            try:
                import diskcache

                # Ensure the cache directory exists
                os.makedirs(CACHE_LOCATION, exist_ok=True)

                # Create a global diskcache Cache instance
                self.cache = diskcache.Cache(CACHE_LOCATION, size_limit=MAX_CACHE_SIZE)
                logger.info(f"Caching enabled at {CACHE_LOCATION}")
            except ImportError as e:
                raise ImportError(
                    "UNITXT_CACHE_LOCATION is set, but diskcache is not installed.\n"
                    "Please install diskcache `pip install diskcache` "
                    "or unset UNITXT_CACHE_LOCATION."
                ) from e
        else:
            self.cache = None  # Disable caching

    def get_or_set(self, key, compute_fn, no_cache=False, refresh=False):
        if not self.cache or no_cache:
            logger.info(f"Bypassing cache for key: {key}")
            return compute_fn()

        if refresh and key in self.cache:
            logger.info(f"Refreshing cache for key: {key}")
            del self.cache[key]

        if key in self.cache:
            logger.info(f"Cache hit for key: {key}")
            return self.cache[key]

        logger.info(f"Cache miss for key: {key}. Computing value...")
        result = compute_fn()
        self.cache[key] = result
        logger.info(f"Stored result in cache for key: {key}")
        return result

    async def async_get_or_set(self, key, compute_fn, no_cache=False, refresh=False):
        if not self.cache or no_cache:
            logger.info(f"Bypassing cache for key: {key}")
            return await compute_fn()

        if refresh and key in self.cache:
            logger.info(f"Refreshing cache for key: {key}")
            del self.cache[key]

        if key in self.cache:
            logger.info(f"Cache hit for key: {key}")
            return self.cache[key]

        logger.info(f"Cache miss for key: {key}. Computing value asynchronously...")
        result = await compute_fn()
        self.cache[key] = result
        logger.info(f"Stored result in cache for key: {key}")
        return result

    def memoize(self, key_func=generate_cache_key, no_cache=False, refresh=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.cache or no_cache:
                    logger.info(f"Bypassing cache for function: {func.__name__}")
                    return func(*args, **kwargs)

                key = key_func(func.__name__, *args, **kwargs)

                if refresh and key in self.cache:
                    logger.info(
                        f"Refreshing cache for function: {func.__name__}, key: {key}"
                    )
                    del self.cache[key]

                if key in self.cache:
                    logger.info(f"Cache hit for function: {func.__name__}, key: {key}")
                    return self.cache[key]

                logger.info(
                    f"Cache miss for function: {func.__name__}, key: {key}. Computing value..."
                )
                result = func(*args, **kwargs)
                self.cache[key] = result
                logger.info(
                    f"Stored result in cache for function: {func.__name__}, key: {key}"
                )
                return result

            return wrapper

        return decorator

    def async_memoize(self, key_func=generate_cache_key, no_cache=False, refresh=False):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if no_cache:
                    logger.info(f"Bypassing cache for async function: {func.__name__}")
                    return await func(*args, **kwargs)

                key = key_func(func.__name__, *args, **kwargs)

                if refresh and key in self.cache:
                    logger.info(
                        f"Refreshing cache for async function: {func.__name__}, key: {key}"
                    )
                    del self.cache[key]

                if key in self.cache:
                    logger.info(
                        f"Cache hit for async function: {func.__name__}, key: {key}"
                    )
                    return self.cache[key]

                logger.info(
                    f"Cache miss for async function: {func.__name__}, key: {key}. Computing value..."
                )
                result = await func(*args, **kwargs)
                self.cache[key] = result
                logger.info(
                    f"Stored result in cache for async function: {func.__name__}, key: {key}"
                )
                return result

            return wrapper

        return decorator
