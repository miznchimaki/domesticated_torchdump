import torch
import torch.utils._pytree as pytree

from numbers import Number
from torchdump.evaluate import (
    calc_diff1,
    calc_diff2,
)


DTYPE_TO_CONFIG_KEY = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}
# Conditionally add dtypes that may not exist in older PyTorch versions
_torch_version = torch.__version__.split('+')[0]
_version_lt = lambda v: next((int(x) < int(y) for x, y in zip(_torch_version.split('.'), v.split('.')) if x != y), False)
if not _version_lt('2.2'):
    DTYPE_TO_CONFIG_KEY[torch.float8_e5m2] = "float8_e5m2"
    DTYPE_TO_CONFIG_KEY[torch.float8_e4m3fn] = "float8_e4m3fn"
    DTYPE_TO_CONFIG_KEY[torch.float8_e5m2fnuz] = "float8_e5m2fnuz"
    DTYPE_TO_CONFIG_KEY[torch.float8_e4m3fnuz] = "float8_e4m3fnuz"
if not _version_lt('2.7'):
    DTYPE_TO_CONFIG_KEY[torch.float8_e8m0fnu] = "float8_e8m0fnu"
if not _version_lt('2.11'):
    DTYPE_TO_CONFIG_KEY[torch.float4_e2m1fn_x2] = "float4_e2m1fn_x2"


def get_dtype_static_threshold(dtype, configs={}):
    """Get per-dtype static threshold from configs.

    Args:
        dtype: The torch.dtype of the tensor being compared.
        configs: The config dict containing dtype_static_thresholds.

    Returns:
        A tuple of (diff1_threshold, diff2_threshold), or (None, None)
        if no per-dtype threshold is configured for this dtype.
    """
    dtype_thresholds = configs.get("dtype_static_thresholds", {})
    if not dtype_thresholds:
        return None, None

    config_key = DTYPE_TO_CONFIG_KEY.get(dtype, None)
    if config_key is None or config_key not in dtype_thresholds:
        return None, None

    dtype_config = dtype_thresholds[config_key]
    diff1 = dtype_config.get("diff1_static_threshold", None)
    diff2 = dtype_config.get("diff2_static_threshold", None)
    return diff1, diff2


def _get_effective_dtype(expect):
    """Extract the effective dtype from a value for per-dtype threshold lookup.

    For a single tensor, returns its dtype directly.
    For containers (list/tuple), returns the dtype of the first tensor found.
    For dict, flattens and finds the first tensor.
    Returns None for non-tensor types.
    """
    if isinstance(expect, torch.Tensor):
        return expect.dtype
    if isinstance(expect, (list, tuple)):
        for item in expect:
            dtype = _get_effective_dtype(item)
            if dtype is not None:
                return dtype
    if isinstance(expect, dict):
        for value in expect.values():
            dtype = _get_effective_dtype(value)
            if dtype is not None:
                return dtype
    return None


def is_custom_tensor_container(obj):
    return hasattr(obj, "_data") and isinstance(getattr(obj, "_data"), torch.Tensor)

def get_per_op_threshold(api, configs={}):
    """Get per-op threshold for a specific API.

    Args:
        api: The API name (e.g., "torch.add", "torch.nn.functional.linear")
        configs: The config dict containing per_op_thresholds

    Returns:
        A tuple of (diff1_threshold, diff2_threshold) for this API,
        or (None, None) if no per-op threshold is defined.

    Matching priority:
        1. Exact match (highest priority)
        2. Longest prefix match (e.g., "torch.nn.functional" wins over "torch.nn")
        3. Longest substring match (fallback)

    For ties at the same length, the lexicographically smallest pattern wins
    to ensure deterministic results regardless of dict iteration order.
    """
    per_op_thresholds = configs.get("per_op_thresholds", {})
    if not per_op_thresholds:
        return None, None

    # Try exact match first (highest priority)
    if api in per_op_thresholds:
        op_threshold = per_op_thresholds[api]
        diff1 = op_threshold.get("diff1_static_threshold", op_threshold.get("diff1", None))
        diff2 = op_threshold.get("diff2_static_threshold", op_threshold.get("diff2", None))
        return diff1, diff2

    # Priority 2: Find best prefix match (longest, with alphabetical tie-breaker)
    best_prefix_pattern = None
    best_prefix_len = -1

    for pattern in per_op_thresholds:
        if api.startswith(pattern + "."):
            match_len = len(pattern)
            # Longer prefix wins; for same length, alphabetically smaller wins
            if match_len > best_prefix_len or (match_len == best_prefix_len and pattern < best_prefix_pattern):
                best_prefix_len = match_len
                best_prefix_pattern = pattern

    if best_prefix_pattern is not None:
        op_threshold = per_op_thresholds[best_prefix_pattern]
        diff1 = op_threshold.get("diff1_static_threshold", op_threshold.get("diff1", None))
        diff2 = op_threshold.get("diff2_static_threshold", op_threshold.get("diff2", None))
        return diff1, diff2

    # Priority 3: Find best substring match (longest, with alphabetical tie-breaker)
    best_substring_pattern = None
    best_substring_len = -1

    for pattern in per_op_thresholds:
        if pattern in api:
            match_len = len(pattern)
            # Longer substring wins; for same length, alphabetically smaller wins
            if match_len > best_substring_len or (match_len == best_substring_len and pattern < best_substring_pattern):
                best_substring_len = match_len
                best_substring_pattern = pattern

    if best_substring_pattern is not None:
        op_threshold = per_op_thresholds[best_substring_pattern]
        diff1 = op_threshold.get("diff1_static_threshold", op_threshold.get("diff1", None))
        diff2 = op_threshold.get("diff2_static_threshold", op_threshold.get("diff2", None))
        return diff1, diff2

    return None, None


class TensorSummary(object):
    def __init__(self, dtype, size, stride, min_val, max_val, mean_val, norm_val, is_meta, requires_grad):
        self.dtype = dtype
        self._size = size
        self.stride = stride
        self.min_val = min_val
        self.max_val = max_val
        self.mean_val = mean_val
        self.norm_val = norm_val
        self.is_meta = is_meta
        self.requires_grad = requires_grad

    # Just to be consistent with the usage of tensor
    def size(self):
        return self._size


@torch.no_grad()
def calc_static_diff(expect, actual, configs={}, api=None):
    if expect is None and actual is None:
        return {}
    if isinstance(expect, TensorSummary) or isinstance(actual, TensorSummary):
        raise RuntimeError("TensorSummary is not supported for calculate diff")
    if is_custom_tensor_container(expect) and is_custom_tensor_container(actual):
        expect, actual = expect._data, actual._data
    # torch.return_type.* == tuple
    assert isinstance(
        actual, type(expect)
    ), "expect and actual must be the same type, got ({}, {}).".format(
        type(expect), type(actual)
    )
    assert get_size(expect) == get_size(
        actual
    ), "expect and actual must be the same length, got ({}, {}).".format(
        get_size(expect), get_size(actual)
    )

    # Get thresholds: global defaults -> per-dtype -> per-op (highest priority)
    diff1_static_threshold = configs.get("diff1_static_threshold", 3e-3)
    diff2_static_threshold = configs.get("diff2_static_threshold", 3e-3)

    # Determine effective dtype for per-dtype threshold lookup
    effective_dtype = _get_effective_dtype(expect)
    if effective_dtype is not None:
        dtype_diff1, dtype_diff2 = get_dtype_static_threshold(effective_dtype, configs)
        if dtype_diff1 is not None:
            diff1_static_threshold = dtype_diff1
        if dtype_diff2 is not None:
            diff2_static_threshold = dtype_diff2

    if api is not None:
        per_op_diff1, per_op_diff2 = get_per_op_threshold(api, configs)
        if per_op_diff1 is not None:
            diff1_static_threshold = per_op_diff1
        if per_op_diff2 is not None:
            diff2_static_threshold = per_op_diff2
    results = {}
    if isinstance(expect, (list, tuple)):
        for i in range(len(expect)):
            # diff is a dict like {"diff1": diff, "diff2": diff2}
            diff = calc_static_diff(expect[i], actual[i], configs, api)
            # only save failed element
            if not check_all_values_pass(diff):
                results[i] = diff
        # every element of list of tuple are all passed
        if len(results) == 0:
            return {"diff1": "PASS", "diff2": "PASS"}
    elif isinstance(expect, dict):
        # flatten leaf elements to list for compare
        return calc_static_diff(pytree.tree_flatten(expect)[0], pytree.tree_flatten(actual)[0], configs, api)
    elif isinstance(expect, torch.Tensor):
        if expect.is_meta:
            assert actual.is_meta == True, f"expect is meta, actual must be meta too"
            results["diff1"] = "PASS"
            results["diff2"] = "PASS"
            return results
        expect = expect.cpu() if not expect.is_cpu else expect
        actual = actual.cpu() if not actual.is_cpu else actual
        results["diff1"] = "PASS"
        results["diff2"] = "PASS"
        if expect.is_complex():
            # for complex, real and imag must both pass
            diff1 = (
                calc_diff1(
                    expect.real.to(torch.float64).numpy(),
                    actual.real.to(torch.float64).numpy(),
                ),
                calc_diff1(
                    expect.imag.to(torch.float64).numpy(),
                    actual.imag.to(torch.float64).numpy(),
                ),
            )
            diff2 = (
                calc_diff2(
                    expect.real.to(torch.float64).numpy(),
                    actual.real.to(torch.float64).numpy(),
                ),
                calc_diff2(
                    expect.imag.to(torch.float64).numpy(),
                    actual.imag.to(torch.float64).numpy(),
                ),
            )
            # for complex, real and imag must both pass
            if diff1[0] > diff1_static_threshold or diff1[1] > diff1_static_threshold:
                results["diff1"] = diff1
            if diff2[0] > diff2_static_threshold or diff2[1] > diff2_static_threshold:
                results["diff2"] = diff2
        else:
            diff1 = calc_diff1(
                expect.to(torch.float64).numpy(), actual.to(torch.float64).numpy()
            )
            diff2 = calc_diff2(
                expect.to(torch.float64).numpy(), actual.to(torch.float64).numpy()
            )
            if diff1 > diff1_static_threshold:
                results["diff1"] = diff1
            if diff2 > diff2_static_threshold:
                results["diff2"] = diff2
    elif isinstance(expect, Number):
        results = calc_static_diff(torch.tensor(expect), torch.tensor(actual), configs, api)
    else:
        raise TypeError(
            "Unsupported type: {}, please add method for this type".format(type(expect))
        )
    return results

def check_all_values_pass(d):
    for value in d.values():
        if isinstance(value, dict):
            for v in value.values():
                if v != "PASS":
                    return False
        else:
            if value != "PASS":
                return False
    return True

def get_size(x):
    size = None
    try:
        if isinstance(x, torch.Tensor):
            size = x.size()
        else:
            size = len(x)
    except TypeError:
        size = 1
    return size

