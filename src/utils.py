from functools import wraps


def requires(*attrs, error_msg=''):
    r"""If attrs aren't defined, raises error."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if any((not hasattr(self, attr) for attr in attrs)):
                raise ValueError(error_msg)
            return fn(self, *args, **kwargs)

        return wrapper

    return decorator