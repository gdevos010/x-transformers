# init helpers


from torch import nn
from functools import wraps

# helpers


def exists(val) -> bool:
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def first(it, default=None):
    return it[0] if len(it) > 0 else default


def is_empty(x) -> bool:
    return len(x) == 0


def cast_tuple(val, depth=1) -> tuple:
    return val if isinstance(val, tuple) else (val,) * depth


def divisible_by(num, den):
    return (num % den) == 0


def maybe(fn=None):
    if not exists(fn):
        fn = identity

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)
