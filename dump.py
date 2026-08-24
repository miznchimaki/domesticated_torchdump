import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch._utils import _rebuild_tensor
from collections import OrderedDict

import functools
import os
import io
import sys
import importlib
import shutil
import copyreg
import dill
import traceback
import pickle
import warnings
from contextlib import contextmanager
import inspect
from typing_extensions import deprecated
import threading
import queue
import re
import types

from .target_op_apis import (
    torch_ops,
    tensor_ops,
    nn_functional_ops,
    torch_fft_ops,
    torch_linalg_ops,
    torch_special_ops,
    nn_module_ops,
    torch_distributed_ops,
    torch_ops_aten_exclude_list,
    mlu_custom_ops,
    tensor_properties,
)
from .utils import (
    add_tensor_info,
    clear_tensor_infos,
    TensorSummary,
    get_min_val,
    get_max_val,
    get_mean_val,
    get_norm_val,
    get_device,
    get_autocast_config,
    get_allow_tf32_config,
    get_optim_outputs,
    import_api_from_str,
    get_logger,
    warning_once,
    __getstate__,
    load_json,
    create_dirs,
    dist_operate,
    stat_tensors_info,
    HIGH,
    MIDDLE,
    LOW,
    get_user_custom_ops,
    HAS_MLU,
    import_module_from_fragmts,
    check_if_include_dtensor_or_faketensor,
)
from .replay import config
from .replay.data_generate import generate_tensor
from .data_collector.data_collector import DataCollector
from .data_collector.task_config import init_task_config
from .grad_stats.grad_monitor import GradientMonitor
from .replay.utils import clone_to_device
from .online_session_part.utils import ApiInfo
from .free_benchmark.grad_checker import is_in_grad_checker
from .autograd_boundary import register_backward_boundary_hook

torch_version_less_2_4=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.4')
torch_version_less_2_11=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.11')
torch_version_less_2_12=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.12') 

_ctx = None

# PYTORCH-12826: skip register hook in grad_fn for not_dump_bwd_ops
not_dump_bwd_ops = ["torch.nn.Identity"]

# Distributed ops can be used in torch.distributed.distributed_c10d._coalescing_manager,
# requires manual maintenance currently.
coalesce_dist_ops = [
    "all_reduce",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "_all_gather_base",
    "_reduce_scatter_base",
]

augment_done = False

api_version = None

# A global flag to ensure that in current process,
# online Dumper instance can only be created once.
FLAG_ONLINE_DUMPER_ONCE = False

logger = get_logger()

# lock used to protect dump_save area
_dump_save_lock = threading.RLock()

# lock used to protect wrapper for torch.autograd.backward
_autograd_backward_lock = threading.RLock()

# A global flag to check if dumper has been started.
# In multi-threaded scenario, users need to ensure by themselves
# that multiple Dumpers do not overlap. When we creates and starts
# a dumper in any thread, it will act on all threads within the
# current process.
DUMPER_ALREADY_IN_USE = False

# A queue saving flags that indicates how many threads is using dumper
THREAD_USING_DUMPER_FLAG_QUEUE = queue.Queue(maxsize=32768)

# A thread local flag indicates if current thread is in __torch_dispatch__
_tls_flag = threading.local()

def is_in_torch_dispatch():
    return getattr(_tls_flag, "IN_TORCH_DISPATCH", False)

@contextmanager
def flag_in_torch_dispatch_context():
    if not hasattr(_tls_flag, "IN_TORCH_DISPATCH"):
        _tls_flag.IN_TORCH_DISPATCH = False
    old_flag = _tls_flag.IN_TORCH_DISPATCH
    _tls_flag.IN_TORCH_DISPATCH = True
    try:
        yield
    finally:
        _tls_flag.IN_TORCH_DISPATCH = old_flag

def flag_in_torch_dispatch_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with flag_in_torch_dispatch_context():
            return func(*args, **kwargs)

    return wrapper

def check_and_set_api_version(cur_version):
    global api_version
    if api_version == 1 - cur_version:
        raise RuntimeError("Mixing the old and new APIs is not supported!")
    if api_version is None:
        api_version = cur_version

def get_context():
    return _ctx

def reset_context():
    global _ctx
    _ctx = None

def clear_context(recover_hijack = True):
    global api_version, DUMPER_ALREADY_IN_USE
    if recover_hijack:
        _ctx.enabled = False
        if _ctx.dump_distributed != "only":
            recover_ops_autograd_bwd_hijack()
    # Block until all other threads finish dumping.
    # If one thread called dumper.stop(), other threads shouldn't stop
    # dumping immediately because they maybe still inside the wrapper
    # of operators. Therefore, we need ensure other threads to finish
    # current operator's dump and then stop it truly.
    THREAD_USING_DUMPER_FLAG_QUEUE.join()
    DUMPER_ALREADY_IN_USE = False
    clear_tensor_infos()
    api_version = None
    return


class Context(object):
    def __init__(
            self,
            task_config,
            output_dir,
            dump_level,
            dump_stack,
            dump_input,
            dump_output,
            op_range,
            op_list,
            skip,
            new_pg,
            cur_rank,
            ranks,
            dump_distributed,
            iters = set(),
            enabled = True
        ):
        assert all(isinstance(a, bool) for a in [enabled, dump_stack, dump_input, dump_output]), \
            f"The types of enabled:{enabled}, dump_stack:{dump_stack}, dump_input:{dump_input} and dump_output:{dump_output} must all be bool."
        assert isinstance(dump_level, int) and dump_level >= LOW and dump_level <= HIGH, \
            f"The type of dump_level param:{dump_level} must be int and the value must be in the range [0, 2]."
        assert isinstance(op_range, list) or isinstance(op_range, tuple), \
            f"The type of op_range:{op_range} param must be list or tuple."
        assert len(op_range) == 0 or len(op_range) == 2, \
            f"op_range:{op_range} must be empty or only contains two elements."
        assert isinstance(op_list, list), f"The type of op_list:{op_list} must be list."
        assert isinstance(skip, int) and skip >= 0, \
            f"The type of skip:{skip} must be int and the value cannot be less than 0."
        assert isinstance(ranks, list), f"The type of ranks:{ranks} must be list."
        assert isinstance(dump_distributed, str), \
            f"The types of dump_distributed:{dump_distributed} must be string."
        assert dump_distributed in ["yes", "no", "only"], f"The value of dump_distributed:{dump_distributed} must be \"yes\", \"no\", or \"only\"."
        if task_config.task == "online":
            global FLAG_ONLINE_DUMPER_ONCE
            assert not FLAG_ONLINE_DUMPER_ONCE, "Online Dumper instance can only be created once in each process."
            FLAG_ONLINE_DUMPER_ONCE = True

            assert dump_level == HIGH, f"dump_level:{dump_level} must be set to HIGH when enable online mode."
            assert dump_input, f"dump_input:{dump_input} must be set to true when enable online mode."
            assert dump_output, f"dump_output:{dump_output} must be set to true when enable online mode."
            assert dump_distributed == "no", f"dump_distributed:{dump_distributed} must be set to \"no\" when enable online mode."
            assert skip == 0, f"Not support to enable skip:{skip} when enable online mode."
        elif task_config.task == "overflow_check":
            assert dump_level == 2, f"when task is overflow_check, dump_level must be 2, but current dump_level is {dump_level}."
            assert dump_input, f"when task is overflow_check, dump_input must be True, but current dump_input is {dump_input}."
            assert dump_output, f"when task is overflow_check, dump_output must be True, but current dump_output is {dump_input}."
            assert len(op_range) == 0, f"when task is overflow_check, op_range must be empty list. but current op_range is {op_range}."
            assert skip == 0, f"when task is overflow_check, skip must be 0. but current skip is {skip}."
        elif task_config.task == "free_benchmark":
            assert dump_level == 2, f"when task is free_benchmark, dump_level must be 2, but current dump_level is {dump_level}."
            assert dump_distributed == "no", f"when task is free_benchmark, dump_distributed must be \"no\", but current dump_distributed is {dump_distributed}."
            assert len(op_range) == 0, f"when task is free_benchmark, op_range must be empty list. but current op_range is {op_range}."
            assert skip == 0, f"when task is free_benchmark, skip must be 0. but current skip is {skip}."

        self._tls_data = threading.local()
        self._dump_input = dump_input if task_config.task != "free_benchmark" else True
        self._dump_input_info = {}
        self._dump_output = dump_output if task_config.task != "free_benchmark" else True
        self._device = get_device()
        self._op_range = [(eval(op) if op == "None" else op) for op in op_range]
        self._op_list = [api.lower() for api in op_list]
        self._pid = os.getpid()
        self._is_in_range = False if op_range else True
        self._first_in_range = False
        self._skip = skip + 1
        self._cnt = 0
        # lock for _is_in_range, _first_in_range, _cnt and _dump_input_info
        self._check_dump_lock = threading.RLock()
        output_dir = os.path.realpath(output_dir)
        self._dump_seqs, self._dump_seqs_lock = {output_dir: [[], []]}, threading.RLock()
        self._fwd_cnt, self._fwd_cnt_lock = -1, threading.RLock()
        if cur_rank is not None:
            self._dump_seqs_path = os.path.join(output_dir, f"rank{cur_rank}", "dump_seqs.pt")
        else:
            self._dump_seqs_path = os.path.join(output_dir, "dump_seqs.pt")
        self._dump_seqs_path_lock = threading.RLock()
        self._cur_iter, self._cur_iter_lock = 0, threading.RLock()
        self._enabled = enabled
        self.iters = iters
        self.task_config = task_config
        self.output_dir = output_dir # used as the nfs_dir when nfs online mode
        self.dump_level = dump_level
        self.dump_stack = dump_stack
        self.dump_any = dump_level > LOW or dump_stack
        self.new_pg = new_pg
        self.cur_rank = cur_rank
        self.ranks = ranks
        self.dump_distributed = dump_distributed
        self.is_online = task_config.task == "online"
        self.online_agent = None
        # stop singal for online mode, won't send data any more once set as True.
        # If set as True, means that current client has disconnected.
        self.online_stopped = False
        if task_config.task == "grad_stats" or self.is_online:
            return
        self.data_collector = DataCollector(task_config)

    @property
    def wrapped_depth(self):
        '''wrapped_depth need be threading local to ensure correct dump logic.
        Different threads sharing the same wrapped_depth would make it be
        unexpected value, because we only dump operators when wrapped_depth
        equals 1.
        '''
        if not hasattr(self._tls_data, "wrapped_depth"):
            self._tls_data.wrapped_depth = 0
        return getattr(self._tls_data, "wrapped_depth")

    @wrapped_depth.setter
    def wrapped_depth(self, value):
        self._tls_data.wrapped_depth = value

    @property
    def is_in_list(self):
        '''is_in_list need be threading local to ensure correct dump logic.
        When one operator is in list in current thread, the operator in
        another thread maybe is not in list. So we cannot make it shared.'''
        return getattr(self._tls_data, "is_in_list", False if self._op_list else True)

    @is_in_list.setter
    def is_in_list(self, value):
        self._tls_data.is_in_list = value

    @property
    def fwd_cnt(self):
        return self._fwd_cnt

    def get_new_fwd_cnt(self):
        '''Add lock for _fwd_cnt in case different ops using same index. Also only
        increasing _fwd_cnt is allowed to ensure it keeps chronological order strictly.
        Do not decrease _fwd_cnt even if dumping nn.Module/Optimizer fails.
        '''
        with self._fwd_cnt_lock:
            self._fwd_cnt += 1
            return self._fwd_cnt

    @property
    def dump_seqs(self):
        return self._dump_seqs

    @property
    def dump_seqs_path(self):
        return self._dump_seqs_path

    @dump_seqs_path.setter
    def dump_seqs_path(self, value):
        with self._dump_seqs_path_lock:
            self._dump_seqs_path = value

    @property
    def cur_iter(self):
        return self._cur_iter

    @property
    def enabled(self):
        is_main_process = self._pid == os.getpid()
        if is_main_process:
            return self._enabled
        elif self._enabled:
            logger.warning("Dumping in subprocess is not supported, "
                "because this behavior will cause the sequence numbers to be messed up.")
        return False

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    # Only used by old dump API, will be deprecated.
    def set_output_dir(self, output_dir):
        self.output_dir = os.path.realpath(output_dir)
        if self.cur_rank is not None:
            self.dump_seqs_path = os.path.join(self.output_dir, f"rank{self.cur_rank}", "dump_seqs.pt")
        else:
            self.dump_seqs_path = os.path.join(self.output_dir, "dump_seqs.pt")

        if self.output_dir in self.dump_seqs.keys():
            return

        # create output dir if output_dir is new
        self._dump_seqs.update({self.output_dir: [[], []]})
        if self.dump_any:
            dist_operate(self, create_dirs, self.output_dir, self.ranks)

    def check_dump_input(self, op_name, wrapped_depth = None, hook_instance = None):
        if self._dump_input and self.check_in_iters():
            if hook_instance is not None:
                hook_instance._op_check_input = True
            return self._check_dump_helper(op_name, wrapped_depth, False)
        return False

    def check_dump_output(self, op_name, wrapped_depth = None, hook_instance = None):
        check_output_in_iters = False
        if hook_instance is None or hook_instance._op_check_input is None:
            check_output_in_iters = self.check_in_iters()
        else:
            check_output_in_iters = hook_instance._op_check_input
        if self._dump_output and check_output_in_iters:
            return self._check_dump_helper(op_name, wrapped_depth, True)
        return False

    def _check_dump_helper(self, op_name, wrapped_depth, is_output):
        def _check_in_range():
            if not self._op_range:
                return
            if self._op_range[0] == self._op_range[1]:
                if self._op_range[0] is None:
                    self._is_in_range = True
                return
            if not self._is_in_range and (op_name == self._op_range[0] \
                or (not self._first_in_range and self._op_range[0] is None)):
                self._is_in_range = True
                self._first_in_range = True
            elif self._is_in_range and (op_name == self._op_range[1] and self._op_range[1]):
                self._is_in_range = False

        def _check_in_list():
            if not self._op_list:
                return
            for api in self._op_list:
                if api in op_name.lower():
                    self.is_in_list = True
                    return
            self.is_in_list = False

        with self._check_dump_lock:
            _check_in_range()
            _check_in_list()
            depth = wrapped_depth if wrapped_depth is not None else _ctx.wrapped_depth
            res = (depth == 1 or (self._op_list and self.is_in_list)) \
                and self._is_in_range and self.is_in_list
            if res and self._skip > 1:
                if self._dump_output and self._dump_input:
                    if is_output and op_name in self._dump_input_info:
                        # There may be nested op call, so recover self._cnt from coresponding input.
                        self._cnt = self._dump_input_info[op_name]
                        self._dump_input_info.pop(op_name)
                    else:
                        self._cnt += 1
                        self._dump_input_info.update({op_name: self._cnt})
                else:
                    self._cnt += 1
                return res and (self._cnt % self._skip == 0)
            return res

    def check_in_iters(self):
        return len(self.iters) == 0 or self.cur_iter in self.iters

    def init_dump_seqs_path(self, dump_seqs_path):
        if not os.path.exists(os.path.dirname(dump_seqs_path)):
            assert 'iter' in dump_seqs_path, "Only lazily create iter subdirectory!"
            os.makedirs(os.path.dirname(dump_seqs_path), exist_ok=True)

    def record_dump_file(self, dump_path, dump_cont, dump_seqs_path, is_run_on_cpu):
        with self._dump_seqs_lock:
            self._dump_seqs[self.output_dir] = [[], []]
            if os.path.exists(dump_seqs_path):
                self._dump_seqs[self.output_dir] = torch.load(dump_seqs_path,  weights_only=False)
            self._dump_seqs[self.output_dir][0].append(dump_path.split('/')[-1])
            tensors_info_dict = {}
            tensors_info_dict["file_name"] = self._dump_seqs[self.output_dir][0][-1]
            tensors_info_dict["is_run_on_cpu"] = is_run_on_cpu
            tensors_info_dict["t_shapes"] = []
            tensors_info_dict["pid"] = os.getpid()
            tensors_info_dict["thread_id"] = threading.get_native_id()
            tensors_info_dict["thread_name"] = threading.current_thread().name
            keys_to_check = ["args", "kwargs", "res", "grad_out", "grad_in"]
            for key in keys_to_check:
                if key in dump_cont.keys():
                   stat_tensors_info(dump_cont[key], tensors_info_dict["t_shapes"])
            tensors_info_dict["t_num"] = len(tensors_info_dict["t_shapes"])
            self._dump_seqs[self.output_dir][1].append(tensors_info_dict)
            torch.save(self._dump_seqs[self.output_dir], dump_seqs_path,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def remove_dump_file_for_item(self, item, dump_seqs_path):
        with self._dump_seqs_lock:
            for idx in range(len(self._dump_seqs[self.output_dir][0])-1, -1, -1):
                if self._dump_seqs[self.output_dir][0][idx] == item:
                    del self._dump_seqs[self.output_dir][0][idx]
                    break
            for idx in range(len(self._dump_seqs[self.output_dir][1])-1, -1, -1):
                if self._dump_seqs[self.output_dir][1][idx]["file_name"] == item:
                    del self._dump_seqs[self.output_dir][1][idx]
                    break
            torch.save(self._dump_seqs[self.output_dir], dump_seqs_path,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def step(self, maybe_create_iter0):
        with self._dump_seqs_lock:
            if maybe_create_iter0 and self.cur_iter == 0 and self.check_in_iters():
                # It's not easy to determine whether current is iter 0 before the first invoking of
                # step() after start(), so we delay the creation of iter0 subdirectory until the first step.
                iter0_dir = os.path.join(self.output_dir, f"iter{self.cur_iter}")
                if self.cur_rank is None:
                    # Here, only need to move pt files to iter0_dir, since there maybe
                    # exist iter directory with dumped data if step() is called before
                    # first start().
                    move_dump_cnt(iter0_dir, self.output_dir, files_only=True)
                else:
                    os.makedirs(iter0_dir, exist_ok=True)
                    iter0_subdir = os.path.join(self.output_dir, f"rank{self.cur_rank}")
                    shutil.move(iter0_subdir, iter0_dir)

            with self._cur_iter_lock:
                self._cur_iter += 1
            if self.cur_rank is not None:
                self.dump_seqs_path = os.path.join(self.output_dir, f"iter{self.cur_iter}",
                    f"rank{self.cur_rank}", "dump_seqs.pt")
            else:
                self.dump_seqs_path = os.path.join(self.output_dir, f"iter{self.cur_iter}",
                    "dump_seqs.pt")

    def online_init(self):
        # tcp has a higher priority than nfs
        if self.task_config.ip_addr:
            from .online_session_part.tcp_agent import TCPAgent
            self.online_agent = TCPAgent(port=self.task_config.port,
                                         ip_addr=self.task_config.ip_addr,
                                         tcp_send_queue_size=self.task_config.tcp_send_queue_size)
        else:
            from .online_session_part.nfs_agent import NFSAgent
            self.online_agent = NFSAgent(nfs_dir=self.output_dir)
        self.online_send("ONLINE_START")

    def online_send(self, data):
        if self.online_stopped:
            return
        name = data.name if isinstance(data, ApiInfo) else data
        logger.info(f"start to send data: {name}")
        self.online_agent.send(data)

    def online_end(self):
        if not self.is_online:
            logger.warning("Current debugging instance was not set as online. "
                          "Use other online instance to call online_terminate() "
                          "if you really need.")
        else:
            try:
                self.online_send("ONLINE_END")
                self.online_stopped = True
                if self.task_config.ip_addr:
                    # disconnect tcp client
                    self.online_agent.can_stop()
            except Exception as e:
                logger.error("Error occurs when calling online_terminate(), " + \
                      f"please ensure having inited online successfully: {e}")


def _filter_unpicklable_items(state):
    """Remove items from state dict that cannot be pickled."""
    if not state:
        return state
    safe = {}
    for k, v in state.items():
        # Fast-path: types and functions are commonly unpicklable
        if isinstance(v, (type, types.FunctionType)):
            continue
        try:
            pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
            safe[k] = v
        except Exception:
            pass
    return safe

def rebuild_parameter(data, requires_grad, hooks, grad, state=None):
    parameter = torch._utils._rebuild_parameter(data, requires_grad, hooks)
    if state is not None:
        torch._utils._set_obj_state(parameter, state)
    if grad is not None:
        parameter.grad = grad
    return parameter

# Modified from torch.nn.Parameter.__reduce_ex__ for the following customizations:
# 1. reserve grad in tensor as needed in Optimizer dump
# 2. preserve Python object state (PT 2.11 semantics)
def wrap_parameter__reduce_ex__(need_grad):
    def parameter__reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        hooks = OrderedDict()
        state = torch._utils._get_obj_state(self)
        if hasattr(self, "_clear_non_serializable_cached_data"):
            self._clear_non_serializable_cached_data()
            state = torch._utils._get_obj_state(self)
        state = _filter_unpicklable_items(state)
        return (
            rebuild_parameter,
            (self.data, self.requires_grad, hooks, self.grad if need_grad else None, state)
        )
    return parameter__reduce_ex__

def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    if config.random_data:
        tensor = generate_tensor(tensor, config.device)

    tensor.requires_grad = requires_grad
    if metadata:
        torch._utils.set_tensor_metadata(tensor, metadata)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    add_tensor_info(tensor)
    return tensor

def rebuild_tensor_v2_with_grad(grad, *args):
    ret = rebuild_tensor_v2(*args)
    ret.grad = grad
    return ret

# Modified from torch.Tensor.__reduce_ex__ for the following customizations:
# 1. conveniently extract tensors for compare after dump
# 2. randomize data in tensor as needed in replay
# 3. reserve grad in tensor as needed in Optimizer dump
# 4. remove the state of tensor to avoid unpicklable error
def wrap_tensor__reduce_ex__(need_grad, hook_instance):
    def wrap_rebuild_tensor_v2(self, func, args):
        if func is torch._utils._rebuild_tensor_v2:
            if need_grad:
                return rebuild_tensor_v2_with_grad, (self.grad,) + args
            else:
                return rebuild_tensor_v2, args
        return func, args
    def tensor__reduce_ex__(self, proto):
        if self.device.type != "cpu":
            hook_instance._is_run_on_cpu = False
        state = torch._utils._get_obj_state(self)
        if type(self) is torch.Tensor and not state:
            # Fast path for regular tensor without Python state.
            func, args = self._reduce_ex_internal(proto)
            func, args = wrap_rebuild_tensor_v2(self, func, args)
            return (func, args)
        if torch.overrides.has_torch_function_unary(self):
            return torch.overrides.handle_torch_function(torch.Tensor.__reduce_ex__, (self,), self, proto)
        func, args = self._reduce_ex_internal(proto)
        func, args = wrap_rebuild_tensor_v2(self, func, args)
        # PT 2.11: clear non-serializable cached data that may reference unpicklable objects
        if hasattr(self, "_clear_non_serializable_cached_data"):
            self._clear_non_serializable_cached_data()
        # Preserve Python object state after clearing cached data (PT 2.11 semantics)
        rebuild_state = torch._utils._get_obj_state(self)
        rebuild_state = _filter_unpicklable_items(rebuild_state)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, rebuild_state))

    return tensor__reduce_ex__

def build_summary_or_random_data(
    dtype,
    size,
    stride,
    min_val,
    max_val,
    mean_val,
    norm_val,
    is_meta,
    requires_grad,
    is_param,
    grad
):
    summary = TensorSummary(dtype, size, stride, min_val, max_val, mean_val, norm_val, is_meta, requires_grad)

    if config.random_data:
        tensor = generate_tensor(summary, config.device)
        if is_param:
            hooks = OrderedDict()
            tensor = torch._utils._rebuild_parameter(tensor, requires_grad, hooks)
        if grad is not None:
            tensor.grad = grad
        return tensor
    add_tensor_info(summary)
    return summary

def wrap_reduce_to_summary(need_grad):
    def reduce_to_summary(self, proto):
        if self.device.type == "meta":
            is_meta = True
            min_val = None
            max_val = None
            mean_val = None
            norm_val = None
        else:
            is_meta = False
            cpu_self = torch._C._TensorBase.cpu(self)
            min_val = get_min_val(cpu_self)
            max_val = get_max_val(cpu_self)
            mean_val = get_mean_val(cpu_self)
            norm_val = get_norm_val(cpu_self)
        is_param = True if isinstance(self, torch.nn.parameter.Parameter) else False
        return (
            build_summary_or_random_data,
            (
                self.dtype,
                tuple(self.size()),
                self.stride(),
                min_val,
                max_val,
                mean_val,
                norm_val,
                is_meta,
                self.requires_grad,
                is_param,
                self.grad if need_grad else None
            )
        )
    return reduce_to_summary

@contextmanager
def dump_guard(ctx, dump_cont, dump_seqs_path, hook_instance):
    ctx.init_dump_seqs_path(dump_seqs_path)
    f0_bak = torch.Tensor.__reduce_ex__
    f1_bak = torch.nn.Parameter.__reduce_ex__
    need_grad = "optim" in dump_cont
    if ctx.dump_level == MIDDLE:
        torch.Tensor.__reduce_ex__ = wrap_reduce_to_summary(need_grad)
        torch.nn.Parameter.__reduce_ex__ = wrap_reduce_to_summary(need_grad)
    elif ctx.dump_level == HIGH:
        if not torch_version_less_2_4:
            copyreg.pickle(torch.Generator, functools.partial(pickle_torch_generator, hook_instance=hook_instance))
        torch.Tensor.__reduce_ex__ = wrap_tensor__reduce_ex__(need_grad, hook_instance)
        torch.nn.Parameter.__reduce_ex__ = wrap_parameter__reduce_ex__(need_grad)
    yield
    torch.Tensor.__reduce_ex__ = f0_bak
    torch.nn.Parameter.__reduce_ex__ = f1_bak

@contextmanager
def disable_dump_context(ctx):
    enabled = ctx.enabled
    ctx.enabled = False
    try:
        yield
    finally:
        ctx.enabled = enabled

def transform_path(full_path, prefix):
    """
    1. truncate the prefix
    2. replace "iter[0-9]+" with "iter0" for the rest
    3. re-add the prefix

    Args:
        full_path: abs path (e.g., "/home/user/iter1/dump_dir/iter123")
        prefix: prefix to be truncated (e.g., "/home/user/iter1")

    Return:
        transformed path (e.g., "/home/user/iter1/dump_dir/iter0")
    """
    if full_path == prefix:
        return full_path
    relative_path = os.path.relpath(full_path, start=prefix)
    transformed = re.sub(r'iter\d+', 'iter0', relative_path)
    result = os.path.join(prefix, transformed)
    return result

def dump_save(ctx, dump_cont, op_name, is_input, hook_instance, need_dump_inner=False):
    if not dump_cont:
        return
    file_suffix = ".input.pt" if is_input else ".output.pt"
    dump_dir = ""
    if is_input:
        hook_instance._op_dump_dir = os.path.dirname(ctx.dump_seqs_path)
        dump_dir = hook_instance._op_dump_dir
    else:
        if hook_instance._op_dump_dir is None:
            # only dump_out
            dump_dir = os.path.dirname(ctx.dump_seqs_path)
        elif "iter" not in os.path.relpath(hook_instance._op_dump_dir, start=ctx.output_dir):
            # if step() is called between dump_input and dump_output, we need update
            # dump_path for output to ensure that both input and output are under
            # 'iter0' directory because input has been moved into it in step().
            dump_dir = transform_path(os.path.dirname(ctx.dump_seqs_path), ctx.output_dir)
        else:
            # use the same dump_dir as input
            dump_dir = hook_instance._op_dump_dir
    dump_path = os.path.join(dump_dir, op_name + file_suffix)
    dump_seqs_path = os.path.join(dump_dir, "dump_seqs.pt")
    
    if check_if_include_dtensor_or_faketensor(dump_cont):
        logger.warning(f"Current task does not support DTensor/FakeTensor now, skip dumping for op: {op_name}{file_suffix}")
        return
    with _dump_save_lock:
        with dump_guard(ctx, dump_cont, dump_seqs_path, hook_instance):
            try:
                torch.save(
                    dump_cont,
                    dump_path,
                    # Work around for the breaking pickle behavior of dill above 0.3.6
                    # Please ref https://github.com/uqfoundation/dill/issues/589.
                    pickle_module=pickle if "optim" in dump_cont else dill,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL
                )
                ctx.record_dump_file(dump_path, dump_cont, dump_seqs_path, hook_instance._is_run_on_cpu)
            except:
                if ctx._op_list and ctx.is_in_list:
                    raise
                args0 = None if len(dump_cont["args"]) == 0 else dump_cont["args"][0]
                if isinstance(args0, torch.nn.Module) or isinstance(args0, torch.optim.Optimizer):
                    # dump the inner op of nn.Module or optim.Optimizer
                    if need_dump_inner:
                        ctx.wrapped_depth -= 1
                    else:
                        raise

                    if os.path.exists(dump_path):
                        os.remove(dump_path)
                    module_type = type(args0)
                    module_name = torch.typename(args0)
                    op_name_ = '.'.join(op_name.split('.')[0: -2])
                    op_instance = import_api_from_str(op_name_, ctx._device)
                    warning_message = f"Failed to dump {module_name}, possibly because some of this object's attributes are unpickable. "
                    if op_instance != module_type:
                        warning_message += f"and {module_name} is subclass of {op_name_}, convert to dump {op_name_}'s inner op(s) instead."
                    else:
                        warning_message += f"convert to dump {op_name_}'s inner op(s) instead."
                    warning_once(warning_message)
                else:
                    raise

class BaseHookTemplate:
    def __init__(self):
        super(BaseHookTemplate, self).__init__()
        self._is_run_on_cpu = True # used for special dump case for cpu tensor(scalar tensor and Generator).
        # guarantee that outputs are saved in the same dir as inputs
        self._op_dump_dir = None
        # guarantee that check_in_iters() for outputs is same as inputs
        self._op_check_input = None

def register_backward_dump_hooks(result, op_name, wrapped_depth, hook_instance, ctx, args, kwargs):
    if not ctx.enabled:
        return False

    bwd_pair_state = {"enabled": False, "grad_out": None, "stack": ""}

    def on_backward_input(grad_out):
        # A background autograd hook can race with Dumper.stop() in no-sync
        # multi-thread runs. Decide once for the whole backward pair so we do
        # not leave only *.bwd.input.pt or only *.bwd.output.pt in dump_seqs.
        bwd_pair_state["enabled"] = ctx.enabled
        if not bwd_pair_state["enabled"]:
            return
        bwd_pair_state["grad_out"] = grad_out
        bwd_pair_state["stack"] = traceback.format_stack() if ctx.dump_stack else ''

    def on_backward_output(grad_in):
        if not bwd_pair_state["enabled"]:
            return
        output_stack = traceback.format_stack() if ctx.dump_stack else ''
        ctx.data_collector.backward_inputs_data_collect(
            bwd_pair_state["grad_out"], op_name, wrapped_depth, ctx, dump_save,
            hook_instance, bwd_pair_state["stack"])
        ctx.data_collector.backward_outputs_data_collect(
            grad_in, op_name, wrapped_depth, ctx, dump_save, hook_instance, output_stack)
        bwd_pair_state["grad_out"] = None
        bwd_pair_state["stack"] = ""

    return register_backward_boundary_hook(
        result, (args, kwargs), on_backward_input, on_backward_output,
        functools.partial(disable_dump_context, ctx))

class HookTemplate(BaseHookTemplate):
    def __init__(self, op, op_name, fwd_cnt):
        super(HookTemplate, self).__init__()
        self._op = op
        self._op_name = op_name + "." + str(fwd_cnt)
        self._skip_bwd_dump = op_name in not_dump_bwd_ops

    # Compared with native nn.Module.__call__ function, we have deleted some unused hooks,
    # supported the scenario where the return value of the operator API is None,
    # and solved the problem of getting grad_fn will report an error in the case shown below:
    # a = torch.randn(3).requires_grad_()
    # with torch.no_grad():
    #   a.view(-1).abs_().grad_fn
    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)

        # Modifying inputs or outputs inplace is not supported by register_full_backward_hook,
        # so we can only use the logic of register_backward_hook, despite some bwd input or
        # output may be lost.
        # TODO(PYTORCH-14349): remove this 'not _ctx.is_online' condition after refactoring codes.
        # TODO(): inner ops in op_list also need register bwd hook
        if not _ctx.is_online \
           and (hasattr(_ctx, "data_collector") and _ctx.data_collector.need_register_bwd_hook()) \
           and torch.is_grad_enabled() \
           and _ctx.wrapped_depth == 1 \
           and not self._skip_bwd_dump:
            register_backward_dump_hooks(result, self._op_name, _ctx.wrapped_depth, self, _ctx, input, kwargs)

        return result

    def forward(self, *args, **kwargs):
        stack = ''
        if _ctx.dump_stack:
            stack = traceback.format_stack()
        if _ctx.is_online:
            # only check dump input is enough
            if _ctx.check_dump_input(self._op_name):
                if check_if_include_dtensor_or_faketensor([args, kwargs]):
                    logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {self._op_name}")
                    res = self._op(*args, **kwargs)
                    return res
                # make a cpu-copy first, to save device memory and in case inputs changed
                # note: keep the original args and kwargs, don't modify them!
                args_copy = clone_to_device(args, "cpu")
                kwargs_copy = clone_to_device(kwargs, "cpu")
                res = self._op(*args, **kwargs)
                if check_if_include_dtensor_or_faketensor(res):
                    logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {self._op_name}")
                    return res
                autocast_config = get_autocast_config(_ctx._device) or \
                                  get_autocast_config('cpu')
                # get the global rank to ensure uniqueness
                rank = dist.get_rank() if dist.is_initialized() else 0
                api_cont = ApiInfo(
                        self._op_name, args_copy, kwargs_copy, res,
                        autocast_config, torch.is_grad_enabled(),
                        get_allow_tf32_config(_ctx._device), rank
                        )
                _ctx.online_send(api_cont)
            else:
                res = self._op(*args, **kwargs)
            return res
        else:
            has_forward_inputs_overflow = _ctx.data_collector.forward_inputs_data_collect(
                    args, kwargs, self._op_name, _ctx, dump_save, None, self, stack)
            if len(args) > 0 and isinstance(args[0], torch.ScriptObject):
                res = self._op(*args[1:], **kwargs)
            else:
                res = self._op(*args, **kwargs)
            if len(args) > 0 and isinstance(args[0], torch.optim.Optimizer):
                res = get_optim_outputs(args[0])
            return _ctx.data_collector.forward_outputs_data_collect(
                    res, self._op_name, _ctx, dump_save, None, self, stack, has_forward_inputs_overflow)

def check_in__coalescing_manager(group):
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    if group in dist.distributed_c10d._world.pg_coalesce_state.keys():
        return True
    return False


def _is_illegal_work(value):
    """Check if value is a _IllegalWork sentinel from coalescing manager."""
    if value is None:
        return False
    try:
        from torch.distributed.distributed_c10d import _IllegalWork
        return isinstance(value, _IllegalWork)
    except (ImportError, AttributeError):
        return False

class HookDistributedOp(BaseHookTemplate):
    def __init__(self, op, op_name, fwd_cnt):
        super(HookDistributedOp, self).__init__()
        self._op = op
        self._op_name = op_name + "." + str(fwd_cnt)

    def __call__(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)

        return result

    def forward(self, *args, **kwargs):
        stack = ''
        if _ctx.dump_stack:
            stack = traceback.format_stack()
        is_batch_isend_irecv = True if self._op_name.split(".")[-2] == "batch_isend_irecv" else False
        batch_isend_irecv_args = []
        input_args = args
        if _ctx.dump_level > LOW:
            if is_batch_isend_irecv:
                for p2pop in args[0]:
                    batch_isend_irecv_args.append(p2pop.tensor)
                input_args = batch_isend_irecv_args

        has_forward_inputs_overflow = _ctx.data_collector.forward_inputs_data_collect(
                input_args, kwargs, self._op_name, _ctx, dump_save, None, self, stack)

        sync_status = True
        if is_batch_isend_irecv:
            handles = self._op(*args, **kwargs)
            is_cpu = False
            if isinstance(args[0], list) and len(args[0]) > 0:
                is_cpu = args[0][0].tensor.is_cpu if isinstance(args[0][0], dist.P2POp) else False
            # is_cpu to avoid "gloo" backend bathc_isend_irecv double wait() will hang issue
            if is_cpu:
                sync_status = False
            else:
                for handle in handles:
                    handle.wait()
        elif self._op_name.split(".")[-2] in coalesce_dist_ops:
            handle = self._op(*args, **kwargs)
            # if user using _coalescing_manager, all_reduce/all_gather_into_tensor/reduce_scatter_tensor
            # cannot call wait(), and may return _IllegalWork.
            if check_in__coalescing_manager(kwargs.get("group")) or _is_illegal_work(handle):
                sync_status = False
            elif kwargs.get("async_op"):
                handle.wait()
        else:
            handle = self._op(*args, **kwargs)
            if self._op_name.split('.')[-2] in ["isend", "irecv"]:
                is_cpu = args[0].is_cpu if isinstance(args[0], torch.Tensor) else False
                # is_cpu to avoid "gloo" backend isend irecv double wait() will hang issue
                # wrapped_depth != 1 to avoid batch_isend_irecv inner isend irecv op do synchronize separately
                if is_cpu or _ctx.wrapped_depth != 1:
                    sync_status = False
                else:
                    handle.wait()
                    sync_status = True
            elif kwargs.get("async_op"):
                handle.wait()

        if sync_status:
            # distributed api (e.g. torch.distributed.all_reduce) are all inplace for input tensor,
            # after execute distributed api and synchronize, still dump the input tensor for output result.
            _ctx.data_collector.forward_outputs_data_collect(
                    input_args, self._op_name, _ctx, dump_save, None, self, stack, has_forward_inputs_overflow)

        if self._op_name.split(".")[-2] == "batch_isend_irecv":
            return handles
        return handle

class BaseDispatchMode(TorchDispatchMode):
    def __init__(self, ctx):
        super().__init__()
        self.ops_prefix = "torch.ops"
        self._ctx = ctx

    # PT 2.11+ compatibility: declare compile semantics explicitly
    # These methods are classmethods in PyTorch 2.11+ TorchDispatchMode infrastructure
    @classmethod
    def is_infra_mode(cls):
        return False

    @classmethod
    def _should_skip_dynamo(cls):
        return True

    @classmethod
    def ignore_compile_internals(cls):
        return False

class OnlineBackwardMode(BaseDispatchMode):
    @flag_in_torch_dispatch_decorator
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_namespace = func.namespace
        op_name = func.__name__.split('.')[0]
        if op_name in torch_ops_aten_exclude_list \
                or op_namespace == "c10d":  # not support distributed ops
            res = func(*args, **kwargs)
            return res

        api_name = self.ops_prefix + "." + op_namespace + "." + op_name + "." + str(self._ctx.get_new_fwd_cnt())
        if self._ctx.check_dump_input(api_name, wrapped_depth=1):
            if check_if_include_dtensor_or_faketensor([args, kwargs]):
                logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {api_name}")
                res = func(*args, **kwargs)
                return res
            # make a cpu-copy first, to save device memory and in case inputs changed
            # note: keep the original args and kwargs, don't modify them!
            args_copy = clone_to_device(args, "cpu")
            kwargs_copy = clone_to_device(kwargs, "cpu")
            res = func(*args, **kwargs)
            if check_if_include_dtensor_or_faketensor(res):
                logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {api_name}")
                return res
            # get the global rank to ensure uniqueness
            rank = dist.get_rank() if dist.is_initialized() else 0
            api_cont = ApiInfo(
                    api_name, args_copy, kwargs_copy, res,
                    None, torch.is_grad_enabled(),
                    get_allow_tf32_config(self._ctx._device), rank
                    )
            self._ctx.online_send(api_cont)
        else:
            res = func(*args, **kwargs)
        return res

class OverflowCheckBackwardMode(BaseDispatchMode):
    @flag_in_torch_dispatch_decorator
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        stack = ''
        op_name = func.__name__.split('.')[0]
        op_namespace = func.namespace
        if op_name in torch_ops_aten_exclude_list:
            res = func(*args, **kwargs)
            return res

        wrapped_depth = 1
        api_name = self.ops_prefix + "." + op_namespace + "." + op_name + "." + str(self._ctx.get_new_fwd_cnt())
        if op_namespace == "c10d":
            # remove torch.classes.c10d.ReduceOp, torch.classes.c10d.ProcessGroup, torch.classes.c10d.Work, etc.
            # these class would be failed to torch.save.
            # these distributed ops are defined in torch/csrc/distributed/c10d/Ops.cpp.
            filter_args = list(filter(lambda x: not isinstance(x, torch.ScriptObject), args))
            has_forward_inputs_overflow = \
                self._ctx.data_collector.forward_inputs_data_collect(
                        filter_args, kwargs, api_name, self._ctx, dump_save, wrapped_depth, func, stack)

            res = func(*args, **kwargs)
            # call Work.wait() (skip _IllegalWork from coalescing manager in PT 2.11+)
            # _IllegalWork means the collective is deferred inside _coalescing_manager;
            # the actual operation hasn't executed yet, so skip output collection too.
            filter_res = filter_args
            if type(res) in [list, tuple]:
                if len(res) > 0 and _is_illegal_work(res[-1]):
                    return res
                filter_res = list(filter(lambda x: not isinstance(x, torch.ScriptObject) and not _is_illegal_work(x), res))
                if len(res) > 0 and not _is_illegal_work(res[-1]):
                    res[-1].wait()
            elif _is_illegal_work(res):
                return res
            else:
                res.wait()
            self._ctx.data_collector.forward_outputs_data_collect(
                    filter_res, api_name, self._ctx, dump_save, wrapped_depth, func, stack, has_forward_inputs_overflow)
            return res

        has_forward_inputs_overflow  = \
            self._ctx.data_collector.forward_inputs_data_collect(
                    args, kwargs, api_name, self._ctx, dump_save, wrapped_depth, func, stack)

        res = func(*args, **kwargs)

        self._ctx.data_collector.forward_outputs_data_collect(
                res, api_name, self._ctx, dump_save, wrapped_depth, func, stack, has_forward_inputs_overflow)
        return res

def is_wrapper_not_supported(is_module_or_optim):
    # TODO(): if online is True or running free_benchmark task,
    # only dump inner ops of nn.Module and
    # Optimizer currently. Need to support dumping nn.Module/Optimizer
    # level together, or allowing user choose to control dump granularity.
    return is_module_or_optim and (_ctx.is_online or _ctx.task_config.task == "free_benchmark")

def op_wrapper_template(op_name, f, template, is_module_or_optim, *args, **kwargs):
    # 1. If _ctx is disabled(once dumper called stop()), we do not wrap op.
    # 2. If some task not support specific ops, we do not wrap op.
    # 3. If bwd grad checker(only for 'free_benchmark' task) is running, we do not wrap inner op.
    # 4. If bwd torch_dispatch(only for 'online'/'overflow_check' task) is running, we do not wrap inner op.
    if (not _ctx.enabled) \
            or is_wrapper_not_supported(is_module_or_optim) \
            or is_in_grad_checker() \
            or is_in_torch_dispatch():
        res = f(*args, **kwargs)
        return res

    if _ctx.wrapped_depth == 0:
        THREAD_USING_DUMPER_FLAG_QUEUE.put("IN_USE")

    _ctx.wrapped_depth += 1
    fwd_cnt = _ctx.get_new_fwd_cnt() if _ctx.wrapped_depth == 1 else _ctx.fwd_cnt
    hook_instance = template(f, op_name, fwd_cnt)

    # [Context wrapped_depth]
    # We always keep the same wrapped_depth before and
    # after calling hook_instance(), no matter how it
    # changes inside hook_instance(). This is to avoid
    # unrecovered wrapped_depth if exception occurs
    # and be catched inside hook_instance().
    orig_wrapped_depth = _ctx.wrapped_depth
    try:
        res = hook_instance(*args, **kwargs)
    finally:
        _ctx.wrapped_depth = orig_wrapped_depth - 1

        if _ctx.wrapped_depth == 0:
            THREAD_USING_DUMPER_FLAG_QUEUE.get()
            THREAD_USING_DUMPER_FLAG_QUEUE.task_done()
    # If f is instance of torch.ScriptMethod, a bound method is being passed,
    # then setting _op to None is done to clear the reference to the object.
    if isinstance(f, torch.ScriptMethod):
        hook_instance._op = None
    return res

def wrap_op(op_name, f, is_module_or_optim=False, template=HookTemplate):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return op_wrapper_template(op_name, f, template, is_module_or_optim, *args, **kwargs)
    # a custom member used as the wrapper mark
    wrapper._td_decorator_mark = "wrap_op"

    # @functools.wraps(f) copies __torch_function__ from the original to
    # the wrapper. This causes TorchFunctionMode (e.g. DeviceContext for
    # torch.device context manager) to intercept the wrapper call, which
    # leads to double invocation: DeviceContext.__torch_function__(wrapper)
    # calls func(*args, **kwargs) → resolves back to wrapper again. Removing
    # __torch_function__ from the wrapper avoids this. The original function
    # call inside the wrapper (self._op) still retains __torch_function__ and
    # will correctly trigger TorchFunctionMode dispatch.
    if hasattr(wrapper, '__torch_function__'):
        delattr(wrapper, '__torch_function__')

    return wrapper

class TensorPropertyDescriptor:
    """Wrapper for tensor properties like H, T, mT, mH (getset_descriptor).

    These properties are not callable methods, so they need special handling.
    When accessed (e.g., tensor.H), this descriptor intercepts the access
    and triggers the dump mechanism.
    """
    def __init__(self, original_descriptor, property_name):
        self._original_descriptor = original_descriptor
        self._property_name = property_name
        self._td_decorator_mark = "wrap_property"

    def __get__(self, obj, objtype=None):
        # If accessed on class (not instance), delegate to original descriptor
        if obj is None:
            return self._original_descriptor.__get__(obj, objtype)

        # Check if _ctx is initialized (may be None after reset_context)
        if _ctx is None:
            return self._original_descriptor.__get__(obj, objtype)

        op_name = f"torch.Tensor.{self._property_name}"

        # Check if dump is enabled - same checks as op_wrapper_template
        if (not _ctx.enabled) \
                or is_in_grad_checker() \
                or is_in_torch_dispatch():
            return self._original_descriptor.__get__(obj, objtype)

        # Handle online mode separately (similar to HookTemplate.forward)
        if _ctx.is_online:
            return self._handle_online_get(obj, objtype, op_name)

        # Handle regular dump mode
        return self._handle_dump_get(obj, objtype, op_name)

    def _handle_online_get(self, obj, objtype, op_name):
        """Handle property access in online mode."""
        # Establish wrapped_depth == 1 before check_dump_input (same as HookTemplate.forward)
        if _ctx.wrapped_depth == 0:
            THREAD_USING_DUMPER_FLAG_QUEUE.put("IN_USE")

        _ctx.wrapped_depth += 1
        fwd_cnt = _ctx.get_new_fwd_cnt() if _ctx.wrapped_depth == 1 else _ctx.fwd_cnt
        full_op_name = op_name + "." + str(fwd_cnt)

        orig_wrapped_depth = _ctx.wrapped_depth
        try:
            if _ctx.check_dump_input(full_op_name):
                # Check for DTensor/FakeTensor before cloning (same as HookTemplate.forward)
                if check_if_include_dtensor_or_faketensor((obj,)):
                    logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {full_op_name}")
                    result = self._original_descriptor.__get__(obj, objtype)
                    return result

                # make a cpu-copy first, to save device memory
                obj_copy = clone_to_device((obj,), "cpu")
                result = self._original_descriptor.__get__(obj, objtype)

                if check_if_include_dtensor_or_faketensor(result):
                    logger.warning(f"Online task does not support DTensor/FakeTensor now, skip sending for op: {full_op_name}")
                    return result

                autocast_config = get_autocast_config(_ctx._device) or get_autocast_config('cpu')
                rank = dist.get_rank() if dist.is_initialized() else 0
                api_cont = ApiInfo(
                    full_op_name, obj_copy, {}, result,
                    autocast_config, torch.is_grad_enabled(),
                    get_allow_tf32_config(_ctx._device), rank
                )
                _ctx.online_send(api_cont)
            else:
                result = self._original_descriptor.__get__(obj, objtype)
        finally:
            _ctx.wrapped_depth = orig_wrapped_depth - 1

            if _ctx.wrapped_depth == 0:
                THREAD_USING_DUMPER_FLAG_QUEUE.get()
                THREAD_USING_DUMPER_FLAG_QUEUE.task_done()

        return result

    def _handle_dump_get(self, obj, objtype, op_name):
        """Handle property access in regular dump mode."""
        if _ctx.wrapped_depth == 0:
            THREAD_USING_DUMPER_FLAG_QUEUE.put("IN_USE")

        _ctx.wrapped_depth += 1
        fwd_cnt = _ctx.get_new_fwd_cnt() if _ctx.wrapped_depth == 1 else _ctx.fwd_cnt
        full_op_name = op_name + "." + str(fwd_cnt)

        # Create a simple hook instance for property access
        hook_instance = TensorPropertyHook(self._property_name, fwd_cnt)

        orig_wrapped_depth = _ctx.wrapped_depth
        try:
            # Collect inputs first (before getting result, for exception consistency)
            stack = ''
            if _ctx.dump_stack:
                stack = traceback.format_stack()

            has_forward_inputs_overflow = _ctx.data_collector.forward_inputs_data_collect(
                (obj,), {}, full_op_name, _ctx, dump_save, None, hook_instance, stack)

            # Get the result from the original property
            result = self._original_descriptor.__get__(obj, objtype)

            # Collect outputs and return the potentially modified result
            result = _ctx.data_collector.forward_outputs_data_collect(
                result, full_op_name, _ctx, dump_save, None, hook_instance, stack, has_forward_inputs_overflow)

            # Register backward hook if needed (similar to HookTemplate.__call__)
            if (hasattr(_ctx, "data_collector") and _ctx.data_collector.need_register_bwd_hook()) \
                    and torch.is_grad_enabled() \
                    and _ctx.wrapped_depth == 1:
                register_backward_dump_hooks(result, full_op_name, _ctx.wrapped_depth, hook_instance, _ctx, (obj,), {})
        finally:
            _ctx.wrapped_depth = orig_wrapped_depth - 1

            if _ctx.wrapped_depth == 0:
                THREAD_USING_DUMPER_FLAG_QUEUE.get()
                THREAD_USING_DUMPER_FLAG_QUEUE.task_done()

        return result

    def __set__(self, obj, value):
        # Delegate to original descriptor, let AttributeError propagate naturally
        return self._original_descriptor.__set__(obj, value)

    def __delete__(self, obj):
        # Delegate to original descriptor, let AttributeError propagate naturally
        return self._original_descriptor.__delete__(obj)


class TensorPropertyHook(BaseHookTemplate):
    """Hook class for tensor property access (H, T, mT, mH)."""
    def __init__(self, property_name, fwd_cnt):
        super(TensorPropertyHook, self).__init__()
        self._property_name = property_name
        self._op_name = f"torch.Tensor.{property_name}.{fwd_cnt}.fwd"
        # For free_benchmark compatibility: provide a callable that mimics property access
        # Properties don't have a real callable, so we create a wrapper
        self._op = lambda tensor: getattr(tensor, property_name)

    def __call__(self, *args, **kwargs):
        # Properties are not callable, this should never be called
        raise RuntimeError(f"TensorPropertyHook should not be called directly for property {self._property_name}")


def wrap_tensor_property(property_name, original_descriptor):
    """Wrap a tensor property (getset_descriptor) for dumping."""
    return TensorPropertyDescriptor(original_descriptor, property_name)

def try_to_serialize(object):
    buffer = io.BytesIO()
    try:
        torch.save(object, buffer, pickle_module=dill, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except:
        return False

def generate_hijacked_class(module, class_name, hijacked_funcs, template=HookTemplate):
    hijacked_class_name = f"Hijacked{class_name}"
    if hijacked_class_name in globals():
        return globals()[hijacked_class_name]

    lines = [
        f"class {hijacked_class_name}:",
        "    def __init__(self, obj):",
        "        self._obj = obj",
        "        self._support_serialization = try_to_serialize(obj)",
        ""
    ]

    for func_name in hijacked_funcs:
        method_full_name = f"{module}.{class_name}.{func_name}"
        lines.append(f"    def {func_name}(self, *args, **kwargs):")
        # dump self._obj to support replay if obj can be serialized.
        lines.append(f"        if self._support_serialization and _ctx.dump_level == HIGH:")
        lines.append(f"            args = (self._obj, ) + args")
        lines.append(f"        return op_wrapper_template('{method_full_name}', self._obj.{func_name}, {template.__name__}, not self._support_serialization, *args, **kwargs)")
        lines.append("")

    lines.append("    def __getattr__(self, func_name):")
    lines.append("        return getattr(self._obj, func_name)")

    code = "\n".join(lines)

    exec(code, globals())
    return globals()[hijacked_class_name]

def wrap_class(cls, hijacked_clas):
    class Wrapper():
        def __init__(self, *args, **kwargs):
            self._raw = cls(*args, **kwargs)
            self._wrapped = hijacked_clas(self._raw)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)
    Wrapper._td_decorator_mark = "wrap_class"

    return Wrapper

def unwrap_hijacked_aten_op(op):
    return getattr(op, "_hijacked_op", getattr(op, "_opPacket", op))

class AtenOpTemplate(torch._ops.OpOverload, BaseHookTemplate):
    def __init__(self, op, op_name):
        # only init BaseHookTemplate, inheriting torch._ops.OpOverload is just to pass the isinstance check.
        BaseHookTemplate.__init__(self)
        self._hijacked_op = op
        self._hijacked_op_name = op_name

    def __getattr__(self, key):
        return getattr(self._hijacked_op, key)

    def __repr__(self):
        return "<AtenOpTemplate(op='{}.{}', overload='{}')>".format(
            *self._schema.name.split("::"), self._overloadname
        )

    def __hash__(self):
        return hash(self._hijacked_op)

    def __eq__(self, other):
        return self._hijacked_op == unwrap_hijacked_aten_op(other)

    def __call__(self, *args, **kwargs):
        return op_wrapper_template(self._hijacked_op_name, self._hijacked_op, HookTemplate, False, *args, **kwargs)

class AtenOpPacketTemplate(torch._ops.OpOverloadPacket, BaseHookTemplate):
    def __init__(self, opPacket, op_name):
        # only init BaseHookTemplate, inheriting torch._ops.OpOverloadPacket is just to pass the isinstance check.
        BaseHookTemplate.__init__(self)
        self._opPacket = opPacket
        self._opPacket_name = op_name
        self._opWrapper = AtenOpTemplate(self._opPacket, self._opPacket_name)

    def __getattr__(self, key):
        try:
            attr = getattr(self._opPacket, key)
        except AttributeError as e:
            raise AttributeError(f"AtenOpPacketTemplate or OpOverloadPacket does not have attribute '{key}'.") from e
        if isinstance(attr, torch._ops.OpOverload) and not isinstance(attr, AtenOpTemplate):
            overload = AtenOpTemplate(attr, self._opPacket_name + '.' + attr._overloadname)
            # need cache the overload object here, in case segment fault of empty reference
            # when importing py::module and getting py::handle in C++ (e.g., getTorchApiFunction)
            setattr(self, key, overload)
            return overload
        else:
            return attr

    def __repr__(self):
        return "<AtenOpPacketTemplate(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    def __hash__(self):
        return hash(self._opPacket)

    def __eq__(self, other):
        return self._opPacket == unwrap_hijacked_aten_op(other)

    def __call__(self, *args, **kwargs):
        return self._opWrapper(*args, **kwargs)

def wrap_aten_op(op_name, opPacket):
    return AtenOpPacketTemplate(opPacket, op_name)

class AtenGetattrTemplate():
    def __init__(self):
        self._getattr_fn = torch.ops.aten.__getattr__

    def __call__(self, *args, **kwargs):
        _opoverloadpacket = self._getattr_fn(*args, **kwargs)
        op_name = args[0]
        if op_name not in torch_ops_aten_exclude_list \
           and callable(getattr(torch.ops.aten, op_name)):
           wrap_op = wrap_aten_op("torch.ops.aten." + op_name, _opoverloadpacket)
           setattr(torch.ops.aten, op_name, wrap_op)
           return wrap_op
        return _opoverloadpacket

def wrap_aten_getattr():
    return AtenGetattrTemplate()

def try_import_dtensor_ops():
    # To support DTensor scenarios and try to avoid throwing errors, need:
    # 1. force import to register sharding rules and strategies for all original aten ops
    # 2. also register items in ShardingPropagator's member dicts for hijacked aten ops
    _dtensor_sharding_propagator = None
    try:
        try:
            import torch.distributed.tensor._ops  # force import all built-in dtensor ops
            _dtensor_sharding_propagator = torch.distributed.tensor.DTensor._op_dispatcher.sharding_propagator
        except:
            try:
                import torch.distributed._tensor.ops  # for torch version below 2.5.0
                _dtensor_sharding_propagator = torch.distributed._tensor.DTensor._op_dispatcher.sharding_propagator
            except:
                import torch.distributed._tensor.ops  # for torch version below and equal to 2.1.0
                _dtensor_sharding_propagator = torch.distributed._tensor.DTensor._propagator
    except:
        logger.error("Failed to import all built-in dtensor ops and dtensor sharding propagator. "
                     "Can not handle all DTensor scenarios, please use torchdump with caution!")
    return _dtensor_sharding_propagator

def hijack_member_of_sharding_propagator(propagator, orig_key, new_key):
    # for more info of ShardingPropagator's member, see:
    # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/distributed/tensor/_sharding_prop.py#L53-L80
    # PT 2.12 introduced op_single_dim_strategy_funcs and op_to_schema_info_for_single_dim_strategy
    # for register_single_dim_strategy (see pytorch PR #177187). Without copying entries from these
    # new dicts, DTensor dispatch fails for ops like _foreach_mul.Tensor that migrated from
    # register_op_strategy to register_single_dim_strategy.
    members = [
        'op_to_rules', 'op_strategy_funcs', 'op_to_schema_info', 'op_to_shape_and_stride_idx',
        'op_single_dim_strategy_funcs', 'op_to_schema_info_for_single_dim_strategy',
    ]
    for member_name in members:
        mem_dict = getattr(propagator, member_name, None)
        if mem_dict is None:
            warning_once(f"{member_name=} not found in DTensor sharding propagator, "
                         "please check it.")
            continue
        if orig_key in mem_dict:
            mem_dict[new_key] = mem_dict[orig_key]

def custom_ops_hijack(custom_ops):
    for _, op_name in enumerate(custom_ops):
        op_name_fragmts = op_name.split('.')
        try:
            prev, cur = import_module_from_fragmts(op_name_fragmts)
            if inspect.isclass(cur) and issubclass(cur, nn.Module):
                # only wrapper once for each op
                if getattr(getattr(cur, "forward"), "_td_decorator_mark", None) != "wrap_op":
                    logger.debug(f"wrapper for custom nn.Module: {cur}")
                    setattr(cur, "forward", wrap_op(op_name, getattr(cur, "forward"), True))
            elif inspect.isclass(cur) and issubclass(cur, torch.optim.Optimizer):
                # only wrapper once for each op
                if getattr(getattr(cur, "step"), "_td_decorator_mark", None) != "wrap_op":
                    logger.debug(f"wrapper for custom Optimizer: {cur}")
                    setattr(cur, "step", wrap_op(op_name, getattr(cur, "step"), True))
                    if cur.__getstate__ is torch.optim.Optimizer.__getstate__:
                        cur.__getstate__ = __getstate__
            else:
                # only wrapper once for each op
                if getattr(cur, "_td_decorator_mark", None) != "wrap_op":
                    logger.debug(f"wrapper for custom op: {cur}")
                    setattr(prev, op_name_fragmts[-1], wrap_op(op_name, cur))
        except:
            logger.warning("Custom op {} cannot be found when hijack in torchdump.".format(op_name))

@functools.lru_cache
def ops_hijack():
    # Pre-populate _device_constructors() lru_cache with original function
    # objects BEFORE wrapping. This ensures that when the original function
    # is called inside a wrapper (self._op), DeviceContext.__torch_function__
    # can recognize it via `original in _device_constructors()` and inject
    # the default device correctly. Without this, the cache might later be
    # populated with wrapper objects, causing the identity check to fail.
    try:
        from torch.utils._device import _device_constructors
        _device_constructors()
    except (ImportError, AttributeError):
        pass

    for attr_name in torch_ops:
        if not hasattr(torch, attr_name):
            continue
        setattr(torch, attr_name, wrap_op("torch." + attr_name, getattr(torch, attr_name)))
        if hasattr(torch._VF, attr_name):
            setattr(torch._VF, attr_name, wrap_op("torch._VF." + attr_name,
                getattr(torch._VF, attr_name)))

    if HAS_MLU:
        tensor_ops.append("mlu")
    for attr_name in tensor_ops:
        if not hasattr(torch.Tensor, attr_name):
            continue
        setattr(torch.Tensor, attr_name, wrap_op("torch.Tensor." + attr_name,
            getattr(torch.Tensor, attr_name)))

    # Wrap tensor properties (H, T, mT, mH) which are getset_descriptors
    for attr_name in tensor_properties:
        if not hasattr(torch.Tensor, attr_name):
            continue
        original_descriptor = getattr(torch.Tensor, attr_name)
        # Only wrap if it's a getset_descriptor (property-like) and not already wrapped
        # getset_descriptor is a built-in type, check by type name
        if type(original_descriptor).__name__ == 'getset_descriptor' \
                and not isinstance(original_descriptor, TensorPropertyDescriptor):
            setattr(torch.Tensor, attr_name, wrap_tensor_property(attr_name, original_descriptor))

    for attr_name in nn_functional_ops:
        if not hasattr(torch.nn.functional, attr_name):
            continue
        setattr(torch.nn.functional, attr_name, wrap_op("torch.nn.functional." + attr_name,
            getattr(torch.nn.functional, attr_name)))

    for attr_name in torch_fft_ops:
        if not hasattr(torch.fft, attr_name):
            continue
        setattr(torch.fft, attr_name, wrap_op("torch.fft." + attr_name,
            getattr(torch.fft, attr_name)))
        alias_name = "fft_" + attr_name
        setattr(torch._C._fft, alias_name, wrap_op("torch._C._fft." + alias_name,
            getattr(torch._C._fft, alias_name)))

    for attr_name in torch_linalg_ops:
        if not hasattr(torch.linalg, attr_name):
            continue
        setattr(torch.linalg, attr_name, wrap_op("torch.linalg." + attr_name,
            getattr(torch.linalg, attr_name)))
        alias_name = "linalg_" + attr_name
        setattr(torch._C._linalg, alias_name, wrap_op("torch._C._linalg." + alias_name,
            getattr(torch._C._linalg, alias_name)))

    for attr_name in torch_special_ops:
        if not hasattr(torch.special, attr_name):
            continue
        setattr(torch.special, attr_name, wrap_op("torch.special." + attr_name,
            getattr(torch.special, attr_name)))
        alias_name = "special_" + attr_name
        setattr(torch._C._special, alias_name, wrap_op("torch._C._special." + alias_name,
            getattr(torch._C._special, alias_name)))

    for attr_name in nn_module_ops:
        if not hasattr(torch.nn, attr_name):
            continue
        module_op = getattr(torch.nn, attr_name)
        setattr(module_op, "forward", wrap_op("torch.nn." + attr_name,
            getattr(module_op, "forward"), True))

    for attr_name in dir(torch.optim):
        optim_op = getattr(torch.optim, attr_name)
        if inspect.isclass(optim_op) and issubclass(optim_op, torch.optim.Optimizer) \
            and 'sparse' not in attr_name.lower():
            setattr(optim_op, "step", wrap_op("torch.optim." + attr_name,
                getattr(optim_op, "step"), True))

    _dtensor_sharding_propagator = try_import_dtensor_ops()
    for attr_name in dir(torch.ops.aten):
        if attr_name not in torch_ops_aten_exclude_list \
            and callable(getattr(torch.ops.aten, attr_name)) \
            and not isinstance(getattr(torch.ops.aten, attr_name), AtenOpPacketTemplate):
            setattr(torch.ops.aten, attr_name, wrap_aten_op("torch.ops.aten." + attr_name,
                getattr(torch.ops.aten, attr_name)))
            if _dtensor_sharding_propagator is not None:
                _op_packet = getattr(torch.ops.aten, attr_name)
                for overload_name in _op_packet.overloads():
                    _op_overload = getattr(_op_packet, overload_name)
                    _hijacked_op = _op_overload._hijacked_op
                    hijack_member_of_sharding_propagator(_dtensor_sharding_propagator, _hijacked_op, _op_overload)

    # dir(torch.ops.aten) can't obtain all operators,
    # so wrap_aten_op can't hijack all operators in torch.ops.aten.
    # When torch.ops.aten.xxx no exist, torch.ops.aten.__getattr__ tries to find it.
    # once a torch.ops.aten.xxx operator is found by torch.ops.aten.__getattr__, then wrap_aten_op hijack it.
    if not isinstance(getattr(torch.ops.aten, "__getattr__"), AtenGetattrTemplate):
        setattr(torch.ops.aten, "__getattr__", wrap_aten_getattr())

    if HAS_MLU:
        # hijack mlu custom_ops defined by torch_mlu, only calling once is enough
        custom_ops_hijack(set(mlu_custom_ops))

@functools.lru_cache
def distributed_ops_hijack():
    for attr_name in torch_distributed_ops:
        if not hasattr(torch.distributed, attr_name):
            continue
        setattr(torch.distributed, attr_name, wrap_op(
            "torch.distributed." + attr_name, getattr(torch.distributed, attr_name), False, HookDistributedOp))
        setattr(torch.distributed.distributed_c10d, attr_name, getattr(torch.distributed, attr_name))

def ops_bwd_hijack_with_dispatch(CustomDispatchMode):
    '''We do not register bwd hook during 'online' or 'overflow_check' tasks.
    In order to catch all bwd ops, Use TorchDispatchMode to hijack
    torch.autograd.backward function and dump Aten-level ops.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Here, we pass _ctx object to CustomDispatchMode. In multi-thread scenario,
            # if one thead is inside decorator(backward) and another thread calls stop()
            # and starts a new dumper in this moment, it would changes global _ctx object.
            # So need keep decorator(backward) in first thread use original same ctx.
            with CustomDispatchMode(_ctx):
                res = func(*args, **kwargs)
                return res

        return wrapper

    torch.autograd.backward = decorator(torch.autograd.backward)

def online_ops_bwd_hijack():
    ops_bwd_hijack_with_dispatch(OnlineBackwardMode)

def overflow_check_ops_bwd_hijack():
    ops_bwd_hijack_with_dispatch(OverflowCheckBackwardMode)

def recover_ops_autograd_bwd_hijack():
    # if different threads call stop() at the same time, maybe cause race
    with _autograd_backward_lock:
        if hasattr(torch.autograd.backward, "__wrapped__"):
            torch.autograd.backward = torch.autograd.backward.__wrapped__

def custom_classes_hijack(custom_classes):
    classes_dict = {}
    for classes in custom_classes:
        parts = classes.split(".")
        if len(parts) < 3:
            logger.error(f"Not support the format of custom classes[{classes}]. \n"
                         f"Expected <module>.<class>.<function>. Skip dumping it.")
            continue
        func = parts[-1]
        class_name = parts[-2]
        module = ".".join(parts[:-2])
        classes_dict.setdefault(module, {}).setdefault(class_name, []).append(func)

    for module, class_info in classes_dict.items():
        for class_name, funcs in class_info.items():
            try:
                hijacked_class = generate_hijacked_class(module, class_name, funcs)
                try:
                    cur = eval(module)
                except Exception:
                    name_fragmts = module.split('.')
                    _, cur = import_module_from_fragmts(name_fragmts)
                # only wrapper once for each class
                if getattr(getattr(cur, class_name), "_td_decorator_mark", None) != "wrap_class":
                    setattr(cur, class_name,
                        wrap_class(getattr(cur, class_name), hijacked_class))
                    logger.debug(f"wrapper for custom class: {module}.{class_name}")
            except Exception as e:
                logger.error(f"Hijack custom classes[{module}.{class_name}] failed: {e}")

def pickle_memory_format(memory_format):
    return str(memory_format).split('.')[1]

def fake_pickle_torch_generator(generator):
    return torch._C.Generator, ()

def pickle_torch_generator(generator, hook_instance):
    if generator.device.type != "cpu":
        hook_instance._is_run_on_cpu = False
    return generator.__reduce__()

def fake_pickle_torch_processgroup(processgroup):
    return torch._C._distributed_c10d.ProcessGroup, (0, 1)

def fake_pickle_torch_stream(stream):
    return torch.Stream, (stream.device, )

def fake_pickle_torch_mlu_stream(stream):
    return torch.mlu.Stream, (stream.device, )

def fake_pickle_torch_cuda_stream(stream):
    return torch.cuda.Stream, (stream.device, )

def fake_pickle_torch_event(event):
    return torch.Event, (event.device, )

def fake_pickle_torch_mlu_event(event):
    return torch.mlu.Event, ()

def fake_pickle_torch_cuda_event(event):
    return torch.cuda.Event, ()

def fake_pickle_bwd_functions(bwd_func):
    return int, ()

def fake_pickle_bwd():
    for name in dir(torch._C._functions):
        if not (name.startswith("__") and name.endswith("__")):
            bwd_fn = getattr(torch._C._functions, name)
            copyreg.pickle(bwd_fn, fake_pickle_bwd_functions)

def augment_torch_serialization():
    global augment_done
    if not augment_done:
        # PyTorch currently do not support serialize these types,
        # we need register the serialization method by ourselves.
        # https://github.com/pytorch/pytorch/issues/56525
        # https://github.com/pytorch/pytorch/issues/43672
        # https://github.com/pytorch/pytorch/issues/71398
        # https://github.com/pytorch/pytorch/issues/76927
        copyreg.pickle(torch.memory_format, pickle_memory_format)
        copyreg.pickle(torch._C._distributed_c10d.ProcessGroup, fake_pickle_torch_processgroup)
        copyreg.pickle(torch.cuda.Stream, fake_pickle_torch_cuda_stream)
        copyreg.pickle(torch.cuda.Event, fake_pickle_torch_cuda_event)
        if HAS_MLU:
            copyreg.pickle(torch.mlu.Stream, fake_pickle_torch_mlu_stream)
            copyreg.pickle(torch.mlu.Event, fake_pickle_torch_mlu_event)
        fake_pickle_bwd()
        if torch_version_less_2_4:
            copyreg.pickle(torch.Generator, fake_pickle_torch_generator)
        else:
            copyreg.pickle(torch.Stream, fake_pickle_torch_stream)
            copyreg.pickle(torch.Event, fake_pickle_torch_event)
        augment_done = True

def check_distributed(process_group, ranks):
    assert isinstance(ranks, list), "The type of ranks must be list."

    dist_check = True
    cur_rank = None
    new_pg = None
    if dist.is_initialized():
        # create new process_group for ctx's attr to avoid influence users' process_group life cycle.
        pg_ranks = dist.get_process_group_ranks(process_group) if process_group else None
        new_pg = dist.new_group(ranks=pg_ranks, backend='gloo')
        cur_rank = dist.get_rank(new_pg)
        if len(ranks) == 0:
            if cur_rank != -1:
                ranks = list(range(dist.get_world_size(new_pg)))
            else:
                # cur_rank == -1 means this process rank is not in process_group ranks,
                # this process's process_group and new_pg are -100, and get_world_size(new_pg) return -1,
                # list(range(-1)) return []
                ranks = list(range(len(pg_ranks)))
        dist_check = cur_rank != -1 and (cur_rank in ranks)
    else:
        ranks = []
    return dist_check, cur_rank, ranks, new_pg

@deprecated("`initialize_dump` is deprecated, Please use `Dumper` instead.")
def initialize_dump(
        enabled = True,
        output_dir = "./dump_dir",
        dump_level = HIGH,
        dump_stack = True,
        dump_input = True,
        dump_output = True,
        op_range = [],
        op_list = [],
        skip = 0,
        process_group = None,
        ranks = [],
        dump_distributed = "yes"
    ):
    check_and_set_api_version(0)
    global _ctx
    if _ctx:
        logger.warning("initialize_dump() has already been called and will not be overwitten. Only dump() "
                "and switch_dump() can overwrite the status of enabled and output_dir.")
        return

    dist_check, cur_rank, ranks, new_pg = check_distributed(process_group, ranks)

    task_config = init_task_config("default", {})

    _ctx = Context(
        task_config,
        output_dir,
        dump_level,
        dump_stack,
        dump_input,
        dump_output,
        op_range,
        op_list,
        skip,
        new_pg,
        cur_rank,
        ranks,
        dump_distributed,
        enabled = enabled
    )

    # Some ranks may not dump, but also need paticipate in creating dump directory
    if _ctx.dump_any:
        dist_operate(_ctx, create_dirs, output_dir, ranks)

    if not dist_check:
        return
    if _ctx.dump_distributed != "only":
        ops_hijack()
        # see note: hijack custom_ops defined by user
        custom_ops, custom_classes = get_user_custom_ops()
        custom_ops_hijack(custom_ops)
        custom_classes_hijack(custom_classes)
    if _ctx.dump_distributed != "no":
        distributed_ops_hijack()

    augment_torch_serialization()

@contextmanager
@deprecated("`dump` is deprecated, Please use `Dumper` instead.")
def dump(
        enabled = True,
        output_dir = "./dump_dir",
        dump_level = HIGH,
        dump_stack = True,
        dump_input = True,
        dump_output = True,
        op_range = [],
        op_list = [],
        skip = 0,
        process_group = None,
        ranks = [],
        dump_distributed = "yes"
    ):
    check_and_set_api_version(0)
    assert isinstance(enabled, bool), f"The type of enabled:{enabled} must be bool."
    assert isinstance(output_dir, str), f"The type of output_dir:{output_dir} must be string."

    prev_enabled = False
    prev_output_dir = None
    if not _ctx:
        initialize_dump(enabled, output_dir, dump_level, dump_stack, dump_input,
            dump_output, op_range, op_list, skip, process_group, ranks, dump_distributed)
    else:
        prev_enabled = _ctx.enabled
        prev_output_dir = _ctx.output_dir
        switch_dump(enabled, output_dir)
    yield
    # If calling dump() to initialize instead of calling initialize_dump(), in the scope out of
    # dump() context block, the status of enabled should be False, and the output_dir should be
    # None. Because dump() is designed to only influence the status of enabled and output_dir
    # within the context block.
    if prev_output_dir is None:
        _ctx.output_dir = None
        _ctx.dump_seqs_path = None
    switch_dump(prev_enabled, prev_output_dir)

@deprecated("`switch_dump` is deprecated, Please use `Dumper` instead.")
def switch_dump(enabled, output_dir = None):
    check_and_set_api_version(0)
    assert isinstance(enabled, bool), f"The type of enabled:{enabled} must be bool."
    if output_dir:
        assert isinstance(output_dir, str), f"The type of output_dir:{output_dir} must be string."

    if not _ctx:
        logger.warning("switch_dump() do nothing. Please initialize dump by calling initialize_dump() or dump() before call switch_dump().")
        return

    if enabled == True and _ctx.output_dir is None:
        assert output_dir is not None, "The current scope's output_dir is None, " \
            "but attempt to enable dump. Please set valid output_dir."

    _ctx.enabled = enabled

    if output_dir is None:
        return

    _ctx.set_output_dir(output_dir)

def parse_iters(input_iters):
    res = set()
    for item in input_iters:
        if isinstance(item, str) and '-' in item:
            beg, end = item.split('-')
            res.update(parse_iters(range(int(beg), int(end))))
        else:
            item = int(item)
            assert item >= 0, f"Iteration number:{item} must greater or equal 0!"
            res.add(item)
    return res

def move_dump_cnt(target_dir, source_dir, files_only=False):
    source_items = os.listdir(source_dir)
    create_dirs(target_dir)
    for item in source_items:
        source_item = os.path.join(source_dir, item)
        if os.path.isfile(source_item) or not files_only:
            shutil.move(source_item, target_dir)

class Dumper(object):
    def __init__(
        self,
        config_path = None,
        output_dir = None,
        dump_level = None,
        process_group = None,
        ranks = None,
        iters = None
    ):
        check_and_set_api_version(1)
        json_config = load_json(config_path)["dump"]
        self.task = json_config.get("task", "default")
        self.gm = None

        dist_check, cur_rank, ranks, new_pg = check_distributed(process_group,
            json_config.get("ranks", []) if ranks is None else ranks)
        self._dist_check = dist_check

        if "TORCHDUMP_DUMP_ITERS" in os.environ:
            env_iters = parse_iters(os.environ["TORCHDUMP_DUMP_ITERS"].split(','))
        else:
            env_iters = set()
        api_iters = parse_iters(json_config.get("iters", []) if iters is None else iters)
        if (len(api_iters) > 0 or iters == []) and len(env_iters) > 0 and api_iters != env_iters:
            logger.warning(f"The dump iters `{api_iters}` received from api or config file "
                f"conflict with `{env_iters}` received from environment. Api or config file "
                "has higher priority than environment.")

        task_config = init_task_config(self.task, json_config)

        self._ctx = Context(
            task_config,
            json_config["output_dir"] if output_dir is None else output_dir,
            json_config["dump_level"] if dump_level is None else dump_level,
            json_config.get("dump_stack", True),
            json_config.get("dump_input", True),
            json_config.get("dump_output", True),
            json_config.get("op_range", []),
            json_config.get("op_list", []),
            json_config.get("skip", 0),
            new_pg,
            cur_rank,
            ranks,
            json_config.get("dump_distributed", "yes"),
            api_iters if len(api_iters) > 0 or iters == [] else env_iters,
            enabled=False, # set enabled to False in constructor, since ops maybe already hijacked
        )
        if self.task == "grad_stats":
            self.gm = GradientMonitor(json_config, self._ctx)
            return

        self._first_started = True

    def start(self):
        if self.gm is not None:
            warning_once(f"Running grad_stats task, Dumper.start shoundn't be called.")
            return

        global _ctx, DUMPER_ALREADY_IN_USE
        if _ctx is self._ctx and DUMPER_ALREADY_IN_USE:
            return

        assert (not DUMPER_ALREADY_IN_USE) or _ctx is self._ctx, \
            "Last instance have not been stopped, " \
            "please stopping previous debugging instance before starting new one!"

        DUMPER_ALREADY_IN_USE = True
        _ctx = self._ctx

        # Some ranks may not dump, but also need paticipate in creating dump directory
        if self._first_started and _ctx.dump_any:
            if _ctx.is_online:
                if not _ctx.task_config.ip_addr:
                    # nfs mode
                    dist_operate(_ctx, create_dirs, _ctx.output_dir, [])
                if self._dist_check:
                    _ctx.online_init()
            elif _ctx.task_config.task == "free_benchmark":
                # "free_benchmark" task only need to create root dir
                dist_operate(_ctx, create_dirs, _ctx.output_dir, [])
            else:
                with _ctx._cur_iter_lock:
                    if _ctx.check_in_iters():
                        dist_operate(_ctx, create_dirs, _ctx.output_dir, _ctx.ranks)
                        # If step() before start(), need to create iter subdirectory
                        if _ctx.cur_iter > 0:
                            target_dir = os.path.join(_ctx.output_dir, f"iter{_ctx.cur_iter}")
                            dist_operate(_ctx, move_dump_cnt, target_dir, _ctx.output_dir)
                    else:
                        # Lazily wait until record_dump_file() to create iter subdirectory.
                        dist_operate(_ctx, create_dirs, _ctx.output_dir, [])

        if not self._dist_check:
            self._first_started = False
            return

        if _ctx.dump_distributed != "only":
            ops_hijack()
            # Note: hijack custom_ops defined by user
            # ops_hijack() is executed at most once, while custom_ops_hijack() for
            # user-defined custom_ops need to be executed in each start(). We need
            # to dynamically check if custom yaml exists and hijack ops in it.
            custom_ops, custom_classes = get_user_custom_ops()
            custom_ops_hijack(custom_ops)
            # custom_classes only support default task
            if _ctx.task_config.task == "default":
                custom_classes_hijack(custom_classes)
            if _ctx.task_config.task =="overflow_check":
                overflow_check_ops_bwd_hijack()
        if _ctx.dump_distributed != "no":
            distributed_ops_hijack()
        if _ctx.is_online:
            online_ops_bwd_hijack()

        augment_torch_serialization()

        # note: set_enabled after augment_torch_serialization has been done
        _ctx.enabled = True
        self._first_started = False

    def stop(self):
        if hasattr(self, "gm") and self.gm is not None:
            return
        if hasattr(self, "_ctx") and _ctx is self._ctx:
            clear_context(self._dist_check)

    def step(self):
        if self.gm is not None:
            warning_once(f"Running grad_stats task, Dumper.step shoundn't be called.")
            return
        maybe_create_iter0 = not self._first_started and self._dist_check and self._ctx.dump_any \
            and not self._ctx.is_online and self._ctx.task_config.task != "free_benchmark"

        self._ctx.step(maybe_create_iter0)

    def monitor(self, model):
        if self.gm and self._dist_check:
            self.gm.monitor(model)

    def online_terminate(self):
        if self.gm is not None:
            return
        if self._ctx.online_stopped:
            return
        if self._dist_check:
            self._ctx.online_end()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        # Skip clearing when Python is shutting down to avoid
        # accessing to objects already be destructed.
        if (sys is not None) and (not sys.is_finalizing()):
            self.stop()
