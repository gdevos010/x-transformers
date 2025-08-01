# init helpers


from functools import partial, wraps

import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn
from torch._tensor import Tensor

# constants

DEFAULT_DIM_HEAD = 64


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


def at_most_one_of(*bools) -> bool:
    return sum(map(int, bools)) <= 1


class always:
    def __init__(self, val) -> None:
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val) -> None:
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals:
    def __init__(self, val) -> None:
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))


def init_zero_(layer) -> None:
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


# tensor helpers


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def max_neg_value(tensor) -> float:
    return -torch.finfo(tensor.dtype).max


def l2norm(t, groups=1) -> Tensor:
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


def softclamp(t, value):
    return (t / value).tanh() * value


def masked_mean(t, mask=None, dim=1):
    if not exists(mask):
        return t.mean(dim=dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp(min=1.0)
    return num / den


def pad_at_dim(t, pad: tuple[int, int], dim=-1, value=0.0):
    if pad == (0, 0):
        return t

    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


LinearNoBias = partial(nn.Linear, bias=False)


# keyword argument helpers


def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in keys)
    return dict(zip(keys, values, strict=False))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    prefix_len = len(prefix)
    kwargs_without_prefix = {
        key[prefix_len:]: value for key, value in kwargs_with_prefix.items()
    }
    return kwargs_without_prefix, kwargs
