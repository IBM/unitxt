import functools
import hashlib
import json
import os

import diskcache

from .logging_utils import get_logger

logger = get_logger()

# Check if caching is enabled via environment variable
CACHE_LOCATION = os.getenv("UNITXT_CACHE_LOCATION")

# Set max cache size to 10GB or the value of env var MAX_CACHE_SIZE
MAX_CACHE_SIZE = os.getenv("MAX_CACHE_SIZE", 10 * 1024**3)

if CACHE_LOCATION:
    # Ensure the cache directory exists
    os.makedirs(CACHE_LOCATION, exist_ok=True)

    # Create a global diskcache Cache instance
    cache = diskcache.Cache(CACHE_LOCATION, size_limit=MAX_CACHE_SIZE)
    logger.info(f"Caching enabled at {CACHE_LOCATION}")
else:
    cache = None  # Disable caching
    logger.info("Caching is disabled (CACHE_LOCATION is not set).")


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


def get_or_set(key, compute_fn, no_cache=False, refresh=False):
    if not cache or no_cache:
        logger.info(f"Bypassing cache for key: {key}")
        return compute_fn()

    if refresh and key in cache:
        logger.info(f"Refreshing cache for key: {key}")
        del cache[key]

    if key in cache:
        logger.info(f"Cache hit for key: {key}")
        return cache[key]

    logger.info(f"Cache miss for key: {key}. Computing value...")
    result = compute_fn()
    cache[key] = result
    logger.info(f"Stored result in cache for key: {key}")
    return result


async def async_get_or_set(key, compute_fn, no_cache=False, refresh=False):
    if not cache or no_cache:
        logger.info(f"Bypassing cache for key: {key}")
        return await compute_fn()

    if refresh and key in cache:
        logger.info(f"Refreshing cache for key: {key}")
        del cache[key]

    if key in cache:
        logger.info(f"Cache hit for key: {key}")
        return cache[key]

    logger.info(f"Cache miss for key: {key}. Computing value asynchronously...")
    result = await compute_fn()
    cache[key] = result
    logger.info(f"Stored result in cache for key: {key}")
    return result


def memoize(key_func=generate_cache_key, no_cache=False, refresh=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cache or no_cache:
                logger.info(f"Bypassing cache for function: {func.__name__}")
                return func(*args, **kwargs)

            key = key_func(func.__name__, *args, **kwargs)

            if refresh and key in cache:
                logger.info(
                    f"Refreshing cache for function: {func.__name__}, key: {key}"
                )
                del cache[key]

            if key in cache:
                logger.info(f"Cache hit for function: {func.__name__}, key: {key}")
                return cache[key]

            logger.info(
                f"Cache miss for function: {func.__name__}, key: {key}. Computing value..."
            )
            result = func(*args, **kwargs)
            cache[key] = result
            logger.info(
                f"Stored result in cache for function: {func.__name__}, key: {key}"
            )
            return result

        return wrapper

    return decorator


def async_memoize(key_func=generate_cache_key, no_cache=False, refresh=False):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if no_cache:
                logger.info(f"Bypassing cache for async function: {func.__name__}")
                return await func(*args, **kwargs)

            key = key_func(func.__name__, *args, **kwargs)

            if refresh and key in cache:
                logger.info(
                    f"Refreshing cache for async function: {func.__name__}, key: {key}"
                )
                del cache[key]

            if key in cache:
                logger.info(
                    f"Cache hit for async function: {func.__name__}, key: {key}"
                )
                return cache[key]

            logger.info(
                f"Cache miss for async function: {func.__name__}, key: {key}. Computing value..."
            )
            result = await func(*args, **kwargs)
            cache[key] = result
            logger.info(
                f"Stored result in cache for async function: {func.__name__}, key: {key}"
            )
            return result

        return wrapper

    return decorator
