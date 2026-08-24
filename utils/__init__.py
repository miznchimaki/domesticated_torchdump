import torch
import importlib

import functools
from contextlib import nullcontext
import json
import yaml
import os
import time
import threading
import random
import numpy as np
from packaging import version
from itertools import chain
import shutil
import numpy as np
import csv
import re
import torch.distributed as dist
from torch.utils._pytree import tree_any
from filelock import FileLock
try:
    from torch.distributed.tensor import DTensor
except:
    from torch.distributed._tensor import DTensor  # for torch version below 2.5.0
from torch._subclasses import FakeTensor

from .calc_diff_utils import *
from .logging_utils import get_logger, warning_once

logger = get_logger()

disable_add_tensor_info = False

CSV_BLACK_LIST = r'^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]'

# The default dump level, all the inputs and outputs including the whole tensors of ops
# are allowed to be dumped.
HIGH = 2

# All the inputs and outputs including tensor summary info of ops are allowed to be dumped.
MIDDLE = 1

# Do not allowed to dump the inputs and outputs of ops.
LOW = 0

try:
    import torch_mlu
    HAS_MLU = True
except ImportError:
    HAS_MLU = False

_tensors_per_pt = []

_custom_op_map = {}
_custom_op_backend_apis = {}
_custom_op_pair_groups = set()
_custom_ops_yaml_lock = threading.RLock()

CUSTOM_OPS_YAML_FILE = "custom_ops.yaml"

def normalize_custom_backend(backend):
    if backend is None:
        return "default"
    backend = backend.lower()
    return "cuda" if backend == "gpu" else backend

def check_custom_op_name(name):
    assert isinstance(name, str) and name, "The name of custom op must be a non-empty string."
    assert os.sep not in name and (os.altsep is None or os.altsep not in name), \
        "The name of custom op cannot contain path separators."

def get_custom_op_qualname(op):
    module = getattr(op, "__module__", None)
    qualname = getattr(op, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return getattr(op, "__name__", str(op))

def get_custom_op_group_name(op):
    return getattr(op, "__qualname__", None) or getattr(op, "__name__", str(op))

def _wrap_registered_custom_op(op, group_name, op_api, backend):
    @functools.wraps(op)
    def wrapper(*args, **kwargs):
        try:
            from torchdump.dump import get_context, HookTemplate, op_wrapper_template
        except Exception:
            return op(*args, **kwargs)
        ctx = get_context()
        if ctx is None or not ctx.enabled:
            return op(*args, **kwargs)
        write_custom_ops_yaml_entry(group_name, op_api, backend, ctx._device)
        return op_wrapper_template(op_api, op, HookTemplate, False, *args, **kwargs)

    wrapper._td_decorator_mark = "register_custom_op"
    wrapper._td_custom_op = op
    return wrapper

def register_custom_op(name=None, backend=None):
    """Register a custom operator using its importable Python API.

    Args:
        name (str, optional): Group name used to pair different backend APIs.
        backend (str, optional): Backend that this implementation runs on.
            Typical values are ``cpu``, ``cuda``, ``gpu`` and ``mlu``.
    """
    if name is not None:
        check_custom_op_name(name)
    if backend is not None:
        assert isinstance(backend, str) and backend, "The backend of custom op must be a non-empty string."
    backend = normalize_custom_backend(backend)

    def decorator(op):
        op_qualname = get_custom_op_qualname(op)
        group_name = name or get_custom_op_group_name(op)
        backend_apis = _custom_op_backend_apis.setdefault(group_name, {})
        registered_api = backend_apis.get(backend)
        assert registered_api is None or registered_api == op_qualname, \
            f"Custom op {group_name} for backend {backend} has already been registered as {registered_api}, got {op_qualname}."
        backend_apis[backend] = op_qualname
        if name is not None:
            _custom_op_pair_groups.add(group_name)
        return _wrap_registered_custom_op(op, group_name, op_qualname, backend)

    return decorator

def _merge_custom_ops_yaml(existing_yaml, new_yaml):
    existing_yaml = existing_yaml if isinstance(existing_yaml, dict) else {}
    merged_yaml = {}
    for device, ops in existing_yaml.items():
        merged_yaml[device] = list(ops or [])
    for device, ops in new_yaml.items():
        merged_ops = merged_yaml.setdefault(device, [])
        for op in ops or []:
            if op not in merged_ops:
                merged_ops.append(op)
    return merged_yaml

def _get_custom_ops_yaml_entry(group_name, api, backend, source_backend, existing_yaml):
    backend = normalize_custom_backend(backend)
    source_backend = backend if backend != "default" else normalize_custom_backend(source_backend)
    backend_apis = _custom_op_backend_apis.get(group_name, {})
    if isinstance(existing_yaml, dict):
        for ops in existing_yaml.values():
            if api in (ops or []):
                return {}

    source_key = source_backend
    target_backend = None
    if isinstance(existing_yaml, dict) and len(existing_yaml) > 0:
        existing_devices = list(existing_yaml.keys())
        if source_key not in existing_devices:
            source_key = existing_devices[0]
        target_backend = next((device for device in existing_devices if device != source_key), None)
    else:
        registered_backends = [backend for backend in backend_apis.keys() if backend != "default"]
        target_backend = next((backend for backend in registered_backends if backend != source_backend), None)
        if target_backend is None and source_backend != "default":
            target_backend = "default"

    if target_backend is None or target_backend == source_key:
        return {}

    target_api = backend_apis.get(target_backend, backend_apis.get("default"))
    if target_api is None:
        if group_name in _custom_op_pair_groups:
            logger.warning(
                f"Cannot find target backend {target_backend} for decorator custom op {group_name}, skip custom ops yaml update."
            )
            return {}
        target_api = api

    return {
        source_key: [api],
        target_backend: [target_api],
    }

def write_custom_ops_yaml_entry(group_name, api, backend=None, source_backend=None):
    yaml_path = os.path.join("./", CUSTOM_OPS_YAML_FILE)
    yaml_dir = os.path.dirname(yaml_path) or "."
    os.makedirs(yaml_dir, exist_ok=True)
    with _custom_ops_yaml_lock, FileLock(yaml_path + ".lock"):
        existing_yaml = {}
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    existing_yaml = yaml.safe_load(f) or {}
            except Exception as exc:
                logger.warning(f"Failed to read existing custom ops yaml {yaml_path}: {exc}")
        custom_ops_yaml = _get_custom_ops_yaml_entry(group_name, api, backend, source_backend, existing_yaml)
        if not custom_ops_yaml:
            return None
        merged_yaml = _merge_custom_ops_yaml(existing_yaml, custom_ops_yaml)
        tmp_path = yaml_path + ".tmp"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(merged_yaml, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, yaml_path)
    return yaml_path

def load_json(config_path=""):
    if not config_path:
        # read default config.json
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir = os.path.dirname(cur_dir)
        config_path = os.path.join(config_dir, "config.json")
    with open(config_path) as f:
        configs = json.load(f)
    return configs

def read_from_custom():
    yaml_path = os.path.join("./", "custom_dump_ops.yaml")
    if not os.path.exists(yaml_path):
        return [], []
    with open(yaml_path, 'r') as f:
        fl = yaml.safe_load(f)
        dev0_custom_ops = fl.get('device0')
        dev1_custom_ops = fl.get('device1')
        if dev0_custom_ops is None:
            dev0_custom_ops = []
        if dev1_custom_ops is None:
            dev1_custom_ops = []
    assert len(dev0_custom_ops) == len(dev1_custom_ops), \
        f"The numbers of custom op of device0:{len(dev0_custom_ops)} and device1:{len(dev1_custom_ops)} must be equal!"
    for op0, op1 in zip(dev0_custom_ops, dev1_custom_ops):
        _custom_op_map[op0] = op1
        _custom_op_map[op1] = op0
    return dev0_custom_ops, dev1_custom_ops

def get_custom_op_map():
    return _custom_op_map

def add_custom_op_map(op_pair):
    assert len(op_pair) == 2
    _custom_op_map[op_pair[0]] = op_pair[1]

def add_tensor_info(tensor_info):
    if not disable_add_tensor_info:
        _tensors_per_pt.append(tensor_info)

def get_tensor_infos():
    global _tensors_per_pt
    res = _tensors_per_pt
    _tensors_per_pt = []
    return res

def clear_tensor_infos():
    global _tensors_per_pt
    _tensors_per_pt = []

def add_time_as_suffix(name):
    if name.endswith('_'):
        return '{}{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
    else:
        return '{}_{}.csv'.format(name, time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))

@torch.no_grad()
def get_min_val(tensor):
    if tensor.numel() == 0:
        return None
    if torch.is_complex(tensor):
        tensor_abs = torch._C._VariableFunctions.abs(tensor)
        return torch._C._TensorBase.item(torch._C._VariableFunctions.min(tensor_abs))
    else:
        return torch._C._TensorBase.item(torch._C._VariableFunctions.min(tensor))

@torch.no_grad()
def get_max_val(tensor):
    if tensor.numel() == 0:
        return None
    if torch.is_complex(tensor):
        tensor_abs = torch._C._VariableFunctions.abs(tensor)
        return torch._C._TensorBase.item(torch._C._VariableFunctions.max(tensor_abs))
    else:
        return torch._C._TensorBase.item(torch._C._VariableFunctions.max(tensor))

@torch.no_grad()
def get_mean_val(tensor):
    if tensor.numel() == 0:
        return None
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        mean_val = torch._C._TensorBase.item(torch._C._VariableFunctions.mean(tensor))
    else:
        mean_val = torch._C._TensorBase.item(torch._C._VariableFunctions.mean(tensor.float()))
    return abs(mean_val) if isinstance(mean_val, complex) else mean_val

@torch.no_grad()
def get_norm_val(tensor):
    if tensor.numel() == 0:
        return None
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        return torch._C._TensorBase.item(torch._C._VariableFunctions.norm(tensor))
    else:
        return torch._C._TensorBase.item(torch._C._VariableFunctions.norm(tensor.float()))

def _get_tensor_device_context(tensor):
    device = tensor.device
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(device)
    if HAS_MLU and device.type == "mlu":
        return torch.mlu.device(device)
    return nullcontext()

@torch.no_grad()
def check_overflow_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 0 or tensor.is_meta:
            return False
        # check inf nan
        with _get_tensor_device_context(tensor):
            is_inf = torch._C._TensorBase.any(torch._C._VariableFunctions.isinf(tensor))
            is_nan = torch._C._TensorBase.any(torch._C._VariableFunctions.isnan(tensor))
            if torch._C._TensorBase.item(is_inf) or torch._C._TensorBase.item(is_nan):
                return True

    return False

def check_overflow_data(data):
    if isinstance(data, float):
        # check inf nan, nan != nan is True
        if data == float('inf') or data == float('-inf') or data != data:
            return True

    return False

def check_overflow(x):
    if isinstance(x, (list, tuple)):
        return any([check_overflow(i) for i in x])
    elif isinstance(x, torch.Tensor):
        return check_overflow_tensor(x)
    else:
        return check_overflow_data(x)

def get_first_tensor(data):
    """get first tensor with requires_grad=True."""
    if isinstance(data, torch.Tensor) and data.requires_grad:
        return data
    elif isinstance(data, dict):
        for value in data.values():
            result = get_first_tensor(value)
            if result is not None:
                return result
    elif isinstance(data, (list, tuple)):
        for item in data:
            result = get_first_tensor(item)
            if result is not None:
                return result
    return None

def get_all_tensors(data):
    """get all tensors with requires_grad=True."""
    result = []
    if isinstance(data, torch.Tensor) and data.requires_grad:
        result.append(data)
    elif isinstance(data, dict):
        for value in data.values():
            result.extend(get_all_tensors(value))
    elif isinstance(data, (list, tuple)):
        for item in data:
            result.extend(get_all_tensors(item))
    return tuple(result)

def get_grad_fn(var):
    var = get_first_tensor(var)
    grad_fn = var.grad_fn if isinstance(var, torch.Tensor) else None
    return grad_fn

def get_device():
    device = "cpu"
    if HAS_MLU and torch.mlu.is_available():
        device = "mlu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        warning_once("Ignore this message if device is cpu. otherwise the device is unsupport, please check!")
    return device

def get_autocast_config(device):
    dtype = None
    # Try unified API first (PyTorch 2.3+)
    try:
        if torch.is_autocast_enabled(device):
            dtype = torch.get_autocast_dtype(device)
    except (TypeError, AttributeError, RuntimeError):
        # Fallback to legacy per-device APIs (PyTorch 2.1-2.3)
        if device == "cuda" and torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
        elif device == "cpu" and torch.is_autocast_cpu_enabled():
            dtype = torch.get_autocast_cpu_dtype()
        elif HAS_MLU and device == "mlu" and torch.mlu.is_autocast_enabled():
            dtype = torch.mlu.get_autocast_dtype()
    return {'device_type': device, 'enabled': True, 'dtype': dtype} if dtype is not None else None

def get_allow_tf32_config(device):
    # only mlu suppport 'custom' ops: torch.dot、torch.addr、torch.addmv、torch.mv
    tf32_cudnn, tf32_matmul, tf32_custom = False, False, False
    if device == "cuda":
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    elif HAS_MLU and device == "mlu":
        tf32_cudnn = torch.backends.cnnl.allow_tf32
        tf32_matmul = torch.backends.mlu.matmul.allow_tf32
        tf32_custom = torch.backends.mlu.custom.allow_tf32
    return {'tf32_cudnn':tf32_cudnn,
            'tf32_matmul':tf32_matmul,
            'tf32_custom':tf32_custom}

def set_allow_tf32_config(device, tf32_dict):
    ctx_cudnn = None
    if device == "cuda":
        ctx_cudnn = torch.backends.cudnn.flags(None, None, None, None, allow_tf32=tf32_dict['tf32_cudnn'])
        torch.backends.cuda.matmul.allow_tf32 = tf32_dict['tf32_matmul']
    elif HAS_MLU and device == "mlu":
        ctx_cudnn = torch.backends.cnnl.flags(None, None, None, None, allow_tf32=tf32_dict['tf32_cudnn'])
        torch.backends.mlu.matmul.allow_tf32 = tf32_dict['tf32_matmul']
        torch.backends.mlu.custom.allow_tf32 = tf32_dict['tf32_custom']
    return ctx_cudnn


def get_optim_outputs(optim):
    return {
        "params": list(chain(*(group["params"] for group in optim.param_groups if "params" in group))),
        "state": optim.state_dict()['state']
    }

# torch.optim.Optimizer.__getstate__ can only save/deepcopy defaults, state and param_groups,
# but other user custom optimizer may have other necessary attributes like overflow_buf in fused adam.
def __getstate__(self):
    return self.__dict__.copy()

def seed_all(seed=1234):
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    except Exception as e:
        logger.warning(f"There is an unexpected error while determinating randomness. {e}")

def create_dirs(output_dir, ranks=[]):
    output_dir = os.path.realpath(output_dir)
    if os.path.exists(output_dir):
        logger.warning("Output directory: {} has already exists and will be overwritten.".format(output_dir))
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for rank in ranks:
        sub_dir = os.path.join(output_dir, f"rank{rank}")
        os.mkdir(sub_dir)

def write_csv(data, filepath, malicious_check=False):
    def csv_value_is_valid(value: str) -> bool:
        if not isinstance(value, str):
            return True
        try:
            # -1.00 or +1.00 should be consdiered as digit numbers
            float(value)
        except ValueError:
            # otherwise, they will be considered as formular injections
            return not bool(re.compile(CSV_BLACK_LIST).search(value))
        return True

    if malicious_check:
        for row in data:
            for cell in row:
                if not csv_value_is_valid(cell):
                    raise RuntimeError(f"Malicious value [{cell}] is not allowed "
                                       f"to be written into the csv: {filepath}.")

    file_path = os.path.realpath(filepath)
    try:
        with open(f"{file_path}", "a", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerows(data)
    except Exception as e:
        logger.error(f'Save csv file "{os.path.basename(file_path)}" failed')
        raise RuntimeError(f"Save csv file {file_path} failed.") from e

def remove_path(path):
    if not os.path.exists(path):
        return
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    except:
        raise RuntimeError(f"Delete {path} failed.")

def _dist_barrier(ctx):
    if getattr(ctx, "new_pg", None) is not None and hasattr(dist, "monitored_barrier"):
        try:
            dist.monitored_barrier(group=ctx.new_pg)
            return
        except ValueError as err:
            if "only implemented for GLOO" not in str(err):
                raise
    dist.barrier(group=ctx.new_pg)

def dist_operate(ctx, func, *args):
    if dist.is_initialized():
        if ctx.cur_rank == ctx.ranks[0]:
            func(*args)
        _dist_barrier(ctx)
    else:
        func(*args)

def stat_tensors_info(x, t_shapes_list):
    if isinstance(x, (tuple, list)) and x:
        for item in x:
            stat_tensors_info(item, t_shapes_list)
    elif isinstance(x, dict) and x:
        for value in x.values():
            stat_tensors_info(value, t_shapes_list)
    elif isinstance(x, torch.Tensor):
        t_shapes_list.append(x.size())
    return

# torch.return_types currently can not be saved by torch.save,
# used to judge whether obj is a torch.return_types,
# may be able to be removed when upgrade pytorch.
def is_structseq(obj):
    cls = type(obj)
    if (
        cls.__base__ is tuple
        and isinstance(getattr(cls, 'n_sequence_fields', None), int)
        and isinstance(getattr(cls, 'n_fields', None), int)
        and isinstance(getattr(cls, 'n_unnamed_fields', None), int)
    ):
        try:
            class subcls(cls):
                pass
        except (
            TypeError,       # CPython
            AssertionError,  # PyPy
        ):
            return True

    return False

def import_api_from_str(api, device):
    op_name_fragmts = api.split(".")
    mod_name = op_name_fragmts[0]
    try:
        mod = importlib.import_module(mod_name)
        for attr_name in op_name_fragmts[1:]:
            if not hasattr(mod, attr_name):
                importlib.import_module(f".{attr_name}", mod.__name__)
            mod = getattr(mod, attr_name)
    except:
        raise ValueError(f"{api} cannot be found for {device}")
    return mod

def check_device(arg, expect):
    def get_device_type(d):
        if isinstance(d, torch.device):
            dt = d.type
        else:
            dt = d.split(":")[0]
        return dt

    return get_device_type(arg) == get_device_type(expect)

def replace_device_arg(args, device):
    if isinstance(args, dict):
        return {k:replace_device_arg(v, device) for k,v in args.items()}
    elif isinstance(args, (list, tuple)):
        return type(args)(replace_device_arg(arg, device) for arg in args)
    elif isinstance(args, torch.device):
        return torch.device(device)
    elif (
          isinstance(args, str)
          and (
               args.startswith("cpu") or
               args.startswith("mlu") or
               args.startswith("cuda")
          )
    ):
        return device
    elif isinstance(args, torch.Generator):
        origin_device = args.device.type
        if check_device(origin_device, device):
            return args
        raise RuntimeError("Does not support change device from {} to {} in torch.Generator".format(origin_device, device))
    else:
        return args

def replay_to_dtype_tensor_handler(obj, dtype):
    if obj.is_floating_point():
        return obj.to(dtype)
    elif obj.is_complex():
        return torch.complex(obj.real.to(dtype), obj.imag.to(dtype))
    return obj

def to_dtype(obj, dtype, tensor_handler=replay_to_dtype_tensor_handler):
    if isinstance(obj, dict):
        return {k:to_dtype(t, dtype, tensor_handler) for k, t in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dtype(t, dtype, tensor_handler) for t in obj)
    elif isinstance(obj, torch.Tensor):
        return tensor_handler(obj, dtype)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(dtype)
    else:
        return obj

def import_module_from_fragmts(fragmts):
    mod_name = fragmts[0]
    mod = importlib.import_module(mod_name)
    prev, cur = None, mod
    for attr_name in fragmts[1:]:
        prev = cur
        if not hasattr(cur, attr_name):
            importlib.import_module(f".{attr_name}", cur.__name__)
        cur = getattr(cur, attr_name)
    return prev, cur

def get_user_custom_ops():
    dev0_custom_ops, dev1_custom_ops = read_from_custom()
    dev0_custom_ops.extend(dev1_custom_ops)
    custom_ops, custom_classes = [], []
    for op in set(dev0_custom_ops):
        try:
            obj = eval(".".join(op.split(".")[:-1]))
            # The obj marked as "wrap_class" is also type of torch.ScriptClass.
            if isinstance(obj, torch.ScriptClass) or getattr(obj, "_td_decorator_mark", None) == "wrap_class":
                custom_classes.append(op)
            else:
                custom_ops.append(op)
        except Exception:
            op_name_fragmts = op.split('.')[:-1]
            try:
                _, cur = import_module_from_fragmts(op_name_fragmts)
                # The obj marked as "wrap_class" is also type of torch.ScriptClass.
                if isinstance(cur, torch.ScriptClass) or getattr(cur, "_td_decorator_mark", None) == "wrap_class":
                    custom_classes.append(op)
                else:
                    custom_ops.append(op)
            except Exception:
                custom_ops.append(op)
    return custom_ops, custom_classes

def check_if_include_dtensor_or_faketensor(data):
    """
    check whether data includes DTensor/FakeTensor using tree_any.

    Args:
        data: input data, can be any data structure
    Returns:
        True only if there is DTensor/FakeTensor obj inside data
    """
    return tree_any(lambda x: isinstance(x, (DTensor, FakeTensor)), data)
