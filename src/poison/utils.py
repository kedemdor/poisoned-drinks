from functools import wraps


class SingletonMeta(type):
    """ Metaclass implementation, shamelessly taken from:
        https://www.datacamp.com/community/tutorials/python-metaclasses"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Cache(metaclass=SingletonMeta):
    """ Cache of function calls"""
    _cache = {}

    @staticmethod
    def function_call_result(f):
        """ Caches the results of the given instance's function call. Assumes the first argument is self."""
        @wraps(f)  # See: https://docs.python.org/3.8/library/functools.html
        def wrapper(*args, **kwds):
            # Devise a hash key. Also makes sure the function parameters are hashable.
            cache_key = tuple(args) + tuple([(kw, val) for kw, val in kwds.items()])
            # If the result of the function call is cached, don't need to perform the function call.
            if cache_key in Cache._cache:
                return Cache._cache[cache_key]
            # Execute the function call & keep track of the result in the cache.
            result = f(*args, **kwds)
            Cache._cache[cache_key] = result
            return result
        return wrapper
