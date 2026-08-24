import torch

import os
import copy
import yaml
import torch.utils._pytree as pytree

from numbers import Number

from torchdump.evaluate import (
    Evaluator,
)
from torchdump.utils import (
    check_overflow,
    TensorSummary,
    get_size,
    check_all_values_pass,
    calc_static_diff,
    is_custom_tensor_container,
    HAS_MLU,
)

def try_import_migration():
    try:
        import torch_mlu
        # if replay on MLU, open gpu_migration
        import torch_mlu.utils.gpu_migration

        # TODO(PYTORCH-13441): need to remove this workround.
        # Currently, use original torch.load to avoid replacing 'cuda' in path
        if hasattr(torch.load, "__wrapped__"):
            torch.load = torch.load.__wrapped__
    except ImportError:
        pass

def rename_privateuse1_backend_for_mlu():
    if HAS_MLU:
        torch.utils.rename_privateuse1_backend("mlu")

def read_ops_standard():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    standard_yaml_path = os.path.join(cur_path, "ops_standard.yaml")
    with open(standard_yaml_path, 'r') as file:
        ops_standard = yaml.safe_load(file)
    return ops_standard
ops_standard = read_ops_standard()

def get_ops_standard(api_str):
    if api_str in ops_standard["binary"]:
        return "binary"
    else:
        return 'default'

def read_from_custom():
    yaml_path = os.path.join("./", "custom_ops.yaml")
    if not os.path.exists(yaml_path):
        return {}, {}

    dev_custom_ops = {}
    with open(yaml_path, 'r') as f:
        fl = yaml.safe_load(f)
        for device, ops in fl.items():
            dev_custom_ops[device] = ops

    if len(dev_custom_ops) == 0:
        return {}, {}

    assert len(dev_custom_ops) == 2, f"The numbers of devices:{len(dev_custom_ops)} must be 2"
    dev0_custom_ops = dev_custom_ops[list(dev_custom_ops.keys())[0]]
    dev1_custom_ops = dev_custom_ops[list(dev_custom_ops.keys())[1]]
    assert len(dev0_custom_ops) == len(dev1_custom_ops), \
           f"The numbers of custom op of device0:{len(dev0_custom_ops)} and device1:{len(dev1_custom_ops)} must be equal!"

    custom_op_map = {}
    for op0, op1 in zip(dev0_custom_ops, dev1_custom_ops):
        custom_op_map[op0] = op1
        custom_op_map[op1] = op0
    custom_ops_dev = {}
    for device, ops in dev_custom_ops.items():
        for op in ops:
            custom_ops_dev[op] = device
    return custom_op_map, custom_ops_dev

def get_custom_api_dict(api, custom_op_map, custom_ops_dev):
    if (extra_api := custom_op_map.get(api, None)) is not None:
        api_dev = custom_ops_dev[api]
        extra_api_dev = custom_ops_dev[extra_api]
        api_dict = {
            api_dev: api,
            extra_api_dev: extra_api,
        }
        return api_dict
    return None

def clone_tensor(x, need_grad=False):
    x_clone = x
    if hasattr(x, 'requires_grad') and x.requires_grad:
        x_clone = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=x.device)
        x_clone.copy_(x)
    if need_grad and hasattr(x, 'grad') and x.grad is not None:
        x_clone.grad = x.grad
    return x_clone

def clone_inputs(inputs, need_grad=False):
    if isinstance(inputs, torch.Tensor):
        return clone_tensor(inputs, need_grad)
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs[key] = clone_inputs(value, need_grad)
        return inputs
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(clone_inputs(elem, need_grad) for elem in inputs)
    return inputs

def update_optim_attrs(optim, func):
    for group in optim.param_groups:
        if "params" not in group:
            continue
        for i, p in enumerate(group["params"]):
            group["params"][i] = func(p)
            optim.state[group["params"][i]] = func(optim.state[p])
            del optim.state[p]
    for key, value in optim.__dict__.items():
        if key == "param_groups" or key == "state":
            continue
        optim.__dict__[key] = func(value)

def clone_tensor_to_device(x, device, need_grad):
    # 1. cuda/mlu->cpu, for static diff check using cpu baseline
    # 2. cpu->cuda/mlu, for Generator device type failure, load to cpu and copy to cuda/mlu
    # MemOverlap::Yes
    if torch._debug_has_internal_overlap(x) == 1:
        arg = x.to(device).detach().requires_grad_(x.requires_grad)
    else:
        # MemOverlap::No or MemOverlap::TooHard
        # we want to remain the same stride info as input as possible, but
        # some overlapping tensor may fail
        arg = torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device=device)
        arg.copy_(x).requires_grad_(x.requires_grad)
    if type(x) is not torch.Tensor:
        arg = type(x)(arg)
    # only for optimizer, remain grad info
    if need_grad and x.grad is not None:
        arg.grad = x.grad.to(device)
    return arg

def clone_to_device(args, device, need_grad=False):
    if isinstance(args, dict):
        args_copy = copy.deepcopy(args)
        for key, value in args_copy.items():
            args_copy[key] = clone_to_device(value, device, need_grad)
        return args_copy
    if isinstance(args, (list, tuple)):
        return type(args)(clone_to_device(arg, device, need_grad) for arg in args)
    if isinstance(args, torch.Tensor):
        # always return args' copy
        return clone_tensor_to_device(args, device, need_grad)
    if isinstance(args, torch.nn.Module):
        args_copy = copy.deepcopy(args)
        return args_copy.to(device)
    if isinstance(args, torch.optim.Optimizer):
        update_optim_attrs(args, lambda x: clone_to_device(x, device, need_grad))
        return args
    return args

def get_device_count(device):
    assert device in ["cpu", "mlu", "cuda"], f"not support device:{device}, device must be [cpu, mlu, cuda]"
    device_count = "torch." + device + ".device_count()"
    return eval(device_count)

def set_current_device(device, device_id):
    assert device in ["cpu", "mlu", "cuda"]
    if device == "cpu":
        # in cpu do nothing
        return
    set_device_cmd = "torch.{}.set_device({})".format(device, device_id)
    eval(set_device_cmd)

# workaround https://github.com/pytorch/pytorch/issues/95711
# to add grad when deepcopy torch.nn.Parameter
def wrap_parameter__deepcopy__(native_deepcopy):
    def parameter__deepcopy__(self, memo):
        res = native_deepcopy(self, memo)
        if self.grad is not None:
            res.grad = self.grad.__deepcopy__(memo)
        memo[id(self)] = res
        return res
    return parameter__deepcopy__

def extract_values(d, name):
    results = {}
    for key, value in d.items():
        if isinstance(value, dict):
            diff_values = extract_values(value, name)
            if diff_values:
                results[key] = diff_values
        else:
            if key == name:
                return value

    if results:
        return results

def calc_dynamic_diff(base, eval1, eval2, configs):
    if eval1 is None and eval2 is None:
        return {}
    if isinstance(eval1, TensorSummary) or isinstance(eval2, TensorSummary):
        raise RuntimeError("TensorSummary is not supported for calculate diff")
    if is_custom_tensor_container(eval1) and is_custom_tensor_container(eval2):
        base, eval1, eval2 = base._data, eval1._data, eval2._data
    assert isinstance(
        eval1, type(eval2)
    ), "eval1 and eval2 must be the same type, got ({}, {}).".format(
        type(eval1), type(eval2)
    )
    assert get_size(eval1) == get_size(
        eval2
    ), "eval1 and eval2 must be the same length, got ({}, {}).".format(
        get_size(eval1), get_size(eval2)
    )

    results = {}
    if isinstance(base, (list, tuple)):
        for i in range(len(base)):
            diff = calc_dynamic_diff(base[i], eval1[i], eval2[i], configs)
            if not check_all_values_pass(diff):
                results[i] = diff
        # every element of list of tuple are all passed
        if len(results) == 0:
            return {"diff1": "PASS", "diff2": "PASS", "diff3": "PASS"}
    elif isinstance(base, dict):
        # flatten leaf elements to list for compare
        return calc_dynamic_diff(
            pytree.tree_flatten(base)[0],
            pytree.tree_flatten(eval1)[0],
            pytree.tree_flatten(eval2)[0],
            configs
        )
    elif isinstance(base, torch.Tensor):
        evaluator = Evaluator(base, eval1, eval2, configs)
        # diff is a dict like {"diff1": [gpu_diff, mlu_diff], "diff2": [gpu_diff, mlu_diff], "diff3": diff}
        diff = evaluator.calc_diff()
        results = evaluator.check_diff(diff)
    elif isinstance(base, Number):
        results = calc_dynamic_diff(torch.tensor(base), torch.tensor(eval1), torch.tensor(eval2), configs)
    else:
        raise TypeError(
            "Unsupported type: {}, please add method for this type".format(
                type(base)
            )
        )
    return results


def replay_on_device_dynamic(
    task,
    device_type,
    cpu_baseline=False,
    configs={},
):
    fwd_results, bwd_results = None, None
    failed_case = None
    try:
        (
            baseline_outputs,
            baseline_grad_inputs,
            mlu_outputs,
            mlu_grad_inputs,
            cuda_outputs,
            cuda_grad_inputs,
        ) = task.run(device_type, cpu_baseline=cpu_baseline, configs=configs)
        configs["standard"] = get_ops_standard(task.api.split(".")[-1])
        fwd_results = calc_dynamic_diff(
            baseline_outputs, mlu_outputs, cuda_outputs, configs
        )
        bwd_results = calc_dynamic_diff(
            baseline_grad_inputs, mlu_grad_inputs, cuda_grad_inputs, configs
        )
    except Exception as e:
        failed_case = [task.rank, task.id, task.api, str(e)]

    res_diff = {}
    if (
        fwd_results is not None
        and bwd_results is not None
        and (not check_all_values_pass(fwd_results)
        or not check_all_values_pass(bwd_results))
    ):
        for diff_name in ["diff1", "diff2", "diff3"]:
            res_diff[f"fwd {diff_name}"] = extract_values(fwd_results, diff_name)
            res_diff[f"bwd {diff_name}"] = extract_values(bwd_results, diff_name)

    return res_diff, failed_case


def replay_on_device_static(
    task,
    device_type,
    cpu_baseline=False,
    overflow_check=False,
    configs={},
):
    fwd_results, bwd_results = None, None
    failed_case = None
    overflow_cmp = {}
    try:
        outputs, expect_outputs, grad_ins, expect_grad_ins = task.run(
            device_type, cpu_baseline=cpu_baseline, configs=configs, overflow_check=overflow_check
        )
        fwd_results = calc_static_diff(expect_outputs, outputs, configs, api=task.api)
        bwd_results = calc_static_diff(expect_grad_ins, grad_ins, configs, api=task.api)
        if overflow_check:
            overflow_cmp["fwd"] = check_overflow(outputs) != check_overflow(expect_outputs)
            if grad_ins is not None:
                overflow_cmp["bwd"] = check_overflow(grad_ins) != check_overflow(expect_grad_ins)
        # del outputs, expect_outputs, grad_ins, expect_grad_ins
        # assert torch.mlu.memory_allocated() == 0
    except Exception as e:
        failed_case = [task.rank, task.id, task.api, str(e)]

    res_diff = {}
    if (
        fwd_results is not None
        and bwd_results is not None
        and (not check_all_values_pass(fwd_results)
        or not check_all_values_pass(bwd_results)
        or len(overflow_cmp) != 0)
    ):
        for diff_name in ["diff1", "diff2"]:
            res_diff[f"fwd {diff_name}"] = extract_values(fwd_results, diff_name)
            res_diff[f"bwd {diff_name}"] = extract_values(bwd_results, diff_name)

    return res_diff, overflow_cmp, failed_case


def replay_on_device_per_task(
    task,
    device_type,
    cpu_baseline=False,
    overflow_check=False,
    configs={},
):
    # save results that is not pass or open overflow_check
    need_save = False

    # dump cases that is not pass or is overflow, and support
    # to replay offline. (this var is used for online only)
    need_dump = False

    # run dynamic accuracy check
    if configs.get("dynamic_accuracy_check", None):
        result = {
            "rank": task.rank,
            "id": task.id,
            "op": task.api,
            "fwd diff1": "",
            "fwd diff2": "",
            "fwd diff3": "",
            "bwd diff1": "",
            "bwd diff2": "",
            "bwd diff3": "",
        }
        res_diff, failed_case = replay_on_device_dynamic(
                task, device_type, cpu_baseline=cpu_baseline, configs=configs
        )
        if len(res_diff) != 0:
            need_save = True
            need_dump = True
            for diff_name in ["diff1", "diff2", "diff3"]:
                result[f"fwd {diff_name}"] = res_diff[f"fwd {diff_name}"]
                result[f"bwd {diff_name}"] = res_diff[f"bwd {diff_name}"]
    # run static accuracy check
    else:
        result = {
            "rank": task.rank,
            "id": task.id,
            "op": task.api,
            "fwd diff1": "",
            "fwd diff2": "",
            "bwd diff1": "",
            "bwd diff2": "",
        }
        if overflow_check:
            result["overflow"] = ""
        res_diff, overflow_cmp, failed_case = replay_on_device_static(
                task, device_type, cpu_baseline=cpu_baseline,
                overflow_check=overflow_check, configs=configs
        )
        if (len(overflow_cmp) == 0 and len(res_diff) != 0) \
                or (len(overflow_cmp) != 0 and overflow_cmp["fwd"]):
            need_dump = True
        if len(res_diff) != 0 or len(overflow_cmp) != 0:
            need_save = True
            for diff_name in ["diff1", "diff2"]:
                result[f"fwd {diff_name}"] = res_diff[f"fwd {diff_name}"]
                result[f"bwd {diff_name}"] = res_diff[f"bwd {diff_name}"]
            if len(overflow_cmp) != 0:
                result["overflow"] = overflow_cmp
    return result, failed_case, need_save, need_dump
