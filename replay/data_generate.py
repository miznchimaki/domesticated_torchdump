import math

import torch
from torch.testing import make_tensor
from torchdump.utils import TensorSummary, get_min_val, get_max_val

random_not_support_dtype_in_mlu = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.bool,
]

def get_type_info(dtype):
    assert isinstance(dtype, torch.dtype), "must be torch.dtype, but is {}".format(dtype)
    if dtype == torch.bool:
        return (0, 2)
    elif dtype.is_floating_point or dtype.is_complex:
        ti = torch.finfo(dtype)
        return (ti.min, ti.max)
    else:
        ti = torch.iinfo(dtype)
        return (ti.min, ti.max)

def is_overlapped_tensor(stride):
    for s in stride:
        if s == 0:
            return True
    return False

def generate_tensor(t, device):
    if isinstance(t, TensorSummary):
        dtype = t.dtype
        size = t._size
        requires_grad = t.requires_grad
        min_val = t.min_val
        max_val = t.max_val
        stride = t.stride
        is_meta = t.is_meta
    elif isinstance(t, torch.Tensor):
        dtype = t.dtype
        size = t.size()
        requires_grad = t.requires_grad
        min_val = get_min_val(t)
        max_val = get_max_val(t)
        stride = t.stride()
        is_meta = t.is_meta
    else:
        return t

    if device is None:
        if isinstance(t, TensorSummary):
            raise RuntimeError(f"must set device for TensorSummary type, but is {t}")
        else:
            device = t.device

    if is_meta:
        return torch.empty_strided(size, stride, dtype=dtype, requires_grad=requires_grad, device="meta")

    dtype_min, dtype_max = get_type_info(dtype)

    # When min_val or max_val is nan, make_tensor will raise "low and high cannot be NaN, but got..."
    if max_val is None or math.isnan(max_val):
        max_val = dtype_max
    if min_val is None or math.isnan(min_val):
        min_val = dtype_min

    cal_stride = [0 for _ in range(len(size))]
    p = 1
    for i in range(len(size)):
        cal_stride[len(size) - 1 - i] = p
        p *= size[len(size)-i-1]

    if min_val == max_val:
        return torch.full(size, min_val, dtype=dtype, device=device).requires_grad_(requires_grad)

    if device == "mlu" and dtype in random_not_support_dtype_in_mlu:
        t = make_tensor(size, dtype=dtype, device="cpu", low=min_val, high=max_val, requires_grad=requires_grad, noncontiguous=False, exclude_zero=False, memory_format=None)
        if cal_stride == list(stride) or is_overlapped_tensor(stride):
            t = t.to(device)
    else:
        t = make_tensor(size, dtype=dtype, device=device, low=min_val, high=max_val, requires_grad=requires_grad, noncontiguous=False, exclude_zero=False, memory_format=None)

    if cal_stride != list(stride) and not is_overlapped_tensor(stride):
        t_with_stride = torch.empty_strided(size, stride, dtype=dtype, device=device)
        t_with_stride.copy_(t).requires_grad_(requires_grad)
        t = t_with_stride
    return t
