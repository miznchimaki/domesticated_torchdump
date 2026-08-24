import copy
import functools
from numbers import Number
from typing import Union
import dill

import torch
from torch.amp import autocast

from torchdump.utils import (
    get_grad_fn,
    get_all_tensors,
    get_optim_outputs,
    __getstate__,
    HAS_MLU,
    get_device,
    get_allow_tf32_config,
    set_allow_tf32_config,
    import_api_from_str,
    get_logger,
    replace_device_arg,
    to_dtype,
)
from torchdump.autograd_boundary import register_backward_boundary_hook

from .utils import (
    clone_inputs,
    update_optim_attrs,
    clone_to_device,
    wrap_parameter__deepcopy__,
)

logger = get_logger()

cpu_not_supported = []

def is_cpu_not_supported(api):
    for ns in cpu_not_supported:
        if ns in api:
            return True
    return False

inplace_op_list = ["__iadd__", "__imul__", "__isub__",
                     "__idiv__", "__itruediv__", "__ifloordiv__",
                     "__imod__", "__iand__", "__ilshift__",
                     "__ior__", "__irshift__", "__ixor__",
                     "__ipow__",
                    ]
inplace_op_list = ["torch.Tensor." + x for x in inplace_op_list]

@functools.lru_cache(None)
def torch_version():
    return torch.__version__

class ReplayTask():
    def __init__(
        self,
        rank: int,
        id: int,
        api: str,
        fwd_inputs: str,
        run_on_cpu_fwd: bool,
        fwd_outputs: str,
        bwd_inputs: str = None,
        run_on_cpu_bwd: bool = False,
        bwd_outputs: str = None,
        api_dict: dict=None,
     ):
        self.rank: int = rank
        self.id: int = id
        self.fwd_inputs: str = fwd_inputs
        self.run_on_cpu_fwd = run_on_cpu_fwd
        self.fwd_outputs: str = fwd_outputs
        self.bwd_inputs: str = bwd_inputs
        self.run_on_cpu_bwd = run_on_cpu_bwd
        self.bwd_outputs: str = bwd_outputs
        self.api = api
        self.api_dict = api_dict
        assert ((self.bwd_inputs is None) == (self.bwd_outputs is None)), \
                f"bwd_input_file must be same with bwd_outputs_files, got bwd_input_file:{self.bwd_inputs},bwd_outputs:{self.bwd_outputs} "
        assert self.fwd_inputs.endswith("fwd.input.pt"), \
                "replay must provide {}.{}.fwd.input.pt, but got {}".format(self.api, str(self.id), self.fwd_inputs )
        self.bwd_access = True if self.bwd_inputs else False

    def info(self):
        return "rank: {}, id: {}, api: {}".format(self.rank, self.id, self.api)

    def _deserialize(self, backend, storage, location):
        if hasattr(torch.serialization, "_deserialize"):
            return torch.serialization._deserialize(backend, storage, location)
        else:
            # For torch 2.1 compatibility
            return torch.UntypedStorage(storage.size(), device=torch.device(location)).copy_(storage, False)

    def map_location(self, is_run_on_cpu, dst_location, storage, src_location):
        if dst_location == 'cpu':
            return storage
        loc = dst_location if dst_location is not None else src_location
        if src_location != 'cpu':
            return self._deserialize(loc.split(":")[0], storage, loc)
        else:
            if is_run_on_cpu:
                return self._deserialize(loc.split(":")[0], storage, loc)
            else:
                return storage

    def get_op_and_inputs(self, map_location):
        map_func = functools.partial(self.map_location, self.run_on_cpu_fwd, map_location)
        content = torch.load(self.fwd_inputs, map_location=map_func, pickle_module=dill)

        args = content["args"]
        kwargs = content["kwargs"]
        autocast_config = content["autocast_config"]
        grad_enable = content["grad_enable"]
        allow_tf32 = content["allow_tf32"]
        self.api = self.api_dict.get(map_location, self.api) if self.api_dict is not None else self.api
        if len(args) > 0 and isinstance(args[0], torch.ScriptObject):
            method = self.api.split(".")[-1]
            op = getattr(args[0], method)
            args = args[1:]
        else:
            op = import_api_from_str(self.api, map_location)
        if isinstance(op, type) and issubclass(op, torch.nn.Module):
            op = op.forward
        if isinstance(op, type) and issubclass(op, torch.optim.Optimizer):
            op = op.step
        logger.debug("op: {}".format(self.api))
        logger.debug("args: {}".format(args))
        logger.debug("kwargs: {}".format(kwargs))
        logger.debug("autocast_config: {}".format(autocast_config))
        logger.debug("grad_enable: {}".format(grad_enable))
        logger.debug("allow_tf32: {}".format(allow_tf32))
        return op, args, kwargs, autocast_config, grad_enable, allow_tf32

    def get_outputs(self, map_location=None):
        content = torch.load(self.fwd_outputs, map_location=map_location, pickle_module=dill)
        outputs = content["res"]
        return outputs

    def get_grad_out(self, map_location=None):
        if self.bwd_inputs is None:
            return None
        map_func = functools.partial(self.map_location, self.run_on_cpu_bwd, map_location)
        content = torch.load(self.bwd_inputs, map_location=map_func, pickle_module=dill)
        grad_out = content["grad_out"]
        logger.debug("grad_out: {}".format(grad_out))
        assert isinstance(grad_out, tuple), f"The type of grad_out:{type(grad_out)} must be tuple."
        return grad_out

    def get_grad_inputs(self, map_location=None):
        if self.bwd_outputs is None:
            return None
        content = torch.load(self.bwd_outputs, map_location=map_location, pickle_module=dill)
        grad_in = content["grad_in"]
        assert isinstance(grad_in, tuple), f"The type of grad_in:{type(grad_in)} must be tuple."
        return grad_in

    def forward(self, op, args, kwargs, autocast_config, grad_enable, allow_tf32):
        is_inplace = True \
                     if (len(args) > 0 and isinstance(args[0], torch.nn.Module) and hasattr(args[0], "inplace") and args[0].inplace is True) or \
                         self.api in inplace_op_list \
                     else False
        args_clone = args
        kwargs_clone = kwargs

        if self.bwd_access and is_inplace:
            args_clone = clone_inputs(args)
            kwargs_clone = clone_inputs(kwargs)

        if len(args_clone) > 0 and isinstance(args_clone[0], torch.optim.Optimizer):
            # clone for every tensor attrs to avoid leaf requires grad variable inplace op error
            if 'differentiable' in args_clone[0].defaults and args_clone[0].defaults['differentiable']:
                update_optim_attrs(args_clone[0], lambda x: clone_inputs(x, True))

        device_type = get_device()

        def wrap_op(args, kwargs):
            def func_with_autocast():
                if autocast_config is None:
                    result = op(*args, **kwargs)
                else:
                    with autocast(**autocast_config):
                        result = op(*args, **kwargs)
                return result

            torch.set_grad_enabled(grad_enable)
            ctx_cudnn = set_allow_tf32_config(device_type, allow_tf32)
            if ctx_cudnn is None:
                result = func_with_autocast()
            else:
                with ctx_cudnn:
                    result = func_with_autocast()

            return result

        outputs = None
        prev_grad = torch.is_grad_enabled()
        prev_allow_tf32 = get_allow_tf32_config(device_type)
        try:
            outputs = wrap_op(args_clone, kwargs_clone)
        except RuntimeError as e:
            error_msg = str(e)
            if "a leaf Variable that requires grad is being used in an in-place operation." in error_msg:
                logger.warning("{} should add to inplace_op_list".format(self.api))
                args_clone = clone_inputs(args)
                kwargs_clone = clone_inputs(kwargs)
                outputs = wrap_op(args_clone, kwargs_clone)
            else:
                raise e
        finally:
            torch.set_grad_enabled(prev_grad)
            set_allow_tf32_config(device_type, prev_allow_tf32)

        if len(args_clone) > 0 and isinstance(args_clone[0], torch.optim.Optimizer):
            outputs = get_optim_outputs(args_clone[0])

        self._last_forward_args = args_clone
        self._last_forward_kwargs = kwargs_clone
        return outputs

    def backward(self, fwd_output, grad_out, args, kwargs):
        def extract_useful_tensors(outputs, grads):
            """Extract pairs of outputs and grads to be used for torch.autograd.backward().

            Args:
                outputs(Tuple[Tensor]): tensors those are extracted from replay's forward
                    output and require grad.
                grads(Tuple[Tensor or None]): grad tensors those are read from dump's
                    backward input pt file.

            Returns:
                (filtered_outputs, filtered_grads)

            Example:
                outputs and grads might exist the following format:
                    CASE1:
                        Inputs:
                            outputs: (y1, y2, y3)
                            grads:   (v1, v2, v3)    # y1's grad_fn need v1, v2 and v3 as input
                        Returns:
                            (y1, y2, y3), (v1, v2, v3)
                    CASE2:
                        Inputs:
                            outputs: (y1, y2, y3)
                            grads:   (v1, None, v3)  # None means y2 doesn't involve autograd
                                                     # when dumping, but y1's grad_fn need it
                                                     # as input. Since this case would throw
                                                     # error in torch.autograd.backward() due
                                                     # to 'None' of grads, we need to remove
                                                     # both None and y2 to handle it. And v2
                                                     # in y1's grad_fn would be filled with
                                                     # 'None' automatically.
                        Returns:
                            (y1, y3), (v1, v3)
                    CASE3:
                        Inputs:
                            outputs: (y1, y2, y3)
                            grads:   (v1,)           # y1's grad_fn only need v1 as input
                        Returns:
                            (y1,), (v1,)
                    CASE4(TODO: Not solved):
                        Inputs:
                            outputs: (y1, y2, y3)
                            grads:   (v1, v3)        # y1's grad_fn need v1 and v3 as input
                        Returns:
                            (y1, y2), (v1, v3)       # we expect (y1, y3), (v1, v3), maybe fail
                                                     # or misreport! see docs of "func_op_f".
            """
            ret_out, ret_grad = [], []
            for out, grad in zip(outputs, grads):
                if grad is not None:
                    assert out.dtype == grad.dtype, f"The dtype of output:{out.dtype} and grad_output:{grad.dtype} is not same, please check!"
                    assert out.shape == grad.shape, f"The shape of output:{out.shape} and grad_output:{grad.shape} is not same, please check!"
                    ret_out.append(out)
                    ret_grad.append(grad)
            return tuple(ret_out), tuple(ret_grad)

        actual_grad_ins = None
        if not self.bwd_access:
            return actual_grad_ins

        def on_backward_input(_grad_out):
            return

        def on_backward_output(grad_in):
            nonlocal actual_grad_ins
            actual_grad_ins = grad_in

        args = getattr(self, "_last_forward_args", args)
        kwargs = getattr(self, "_last_forward_kwargs", kwargs)
        try:
            if register_backward_boundary_hook(fwd_output, (args, kwargs), on_backward_input, on_backward_output):
                out_tensors = get_all_tensors(fwd_output)
                legal_out, legal_grad_out = extract_useful_tensors(out_tensors, grad_out)
                torch.autograd.backward(legal_out, legal_grad_out)
            else:
                logger.warning(f"{self.info()} have *.bwd.input.pt but grad_fn is None, since does not compare output of backward.")
        finally:
            self._last_forward_args = None
            self._last_forward_kwargs = None

        return actual_grad_ins

    # for dynamic check, run fp64 golden baseline
    # for static check, run cpu baseline only when cpu_baseline is set
    def run_offline_baseline(
        self,
        op,
        args,
        kwargs,
        grad_out,
        autocast_config,
        grad_enable,
        allow_tf32,
        baseline_device='cuda',
        dynamic_check=False):
        args0_copy = None
        if isinstance(args[0], torch.nn.Module):
            args0_copy = copy.deepcopy(args[0]).to(baseline_device)
        elif isinstance(args[0], torch.optim.Optimizer):
            # workaround https://github.com/pytorch/pytorch/issues/95711
            # to add grad when deepcopy torch.nn.Parameter
            bak = torch.nn.Parameter.__deepcopy__
            torch.nn.Parameter.__deepcopy__ = wrap_parameter__deepcopy__(bak)
            if not torch.typename(args[0]).startswith('torch.optim.') \
                and type(args[0]).__getstate__ is torch.optim.Optimizer.__getstate__:
                type(args[0]).__getstate__ = __getstate__
            args0_copy = clone_to_device(copy.deepcopy(args[0]), baseline_device, True)
            if type(args[0]).__getstate__ is __getstate__:
                type(args[0]).__getstate__ = torch.optim.Optimizer.__getstate__
            torch.nn.Parameter.__deepcopy__ = bak
        else:
            # for custom ops
            api = self.api_dict.get(baseline_device, self.api) if self.api_dict is not None else self.api
            if api != self.api:
                op = import_api_from_str(api, baseline_device)
        if args0_copy is None:
            args_copy = clone_to_device(args, baseline_device)
        else:
            args_copy = clone_to_device(args[1:], baseline_device)
            args_copy = (args0_copy, ) + args_copy
        kwargs_copy = clone_to_device(kwargs, baseline_device)
        args_copy = replace_device_arg(args_copy, baseline_device)
        kwargs_copy = replace_device_arg(kwargs_copy, baseline_device)
        grad_out_copy = clone_to_device(grad_out, baseline_device)
        autocast_config_copy = copy.deepcopy(autocast_config)
        if autocast_config_copy is not None:
            autocast_config_copy["device_type"] = baseline_device
            # Pytorch2.1 CPU AutoCastMode only support bfloat16
            if torch_version().startswith("2.1") and baseline_device == "cpu":
                autocast_config_copy["dtype"] = torch.bfloat16

        # run fp64 baseline
        if dynamic_check:
            # close autocast
            autocast_config_copy = None
            args_copy = to_dtype(args_copy, dtype=torch.float64)
            kwargs_copy = to_dtype(kwargs_copy, dtype=torch.float64)
            grad_out_copy = to_dtype(grad_out_copy, dtype=torch.float64)

        try:
            baseline_outputs = self.forward(
                op, args_copy, kwargs_copy, autocast_config_copy, grad_enable, allow_tf32)
        # raise "'op' not implemented for 'dtype'", then input/weight cast to float.
        except RuntimeError as e:
            # for old pytorch version, some cpu ops not support half or bfloat16 and
            # cast to float32. It is an old behavior and I just remain it as before.
            # For dynamic check, we just raise error because it only runs fp64.
            if not dynamic_check and "not implemented for" in str(e):
                logger.warning(str(e) + ", and cast to torch.float to replay.")
                autocast_config_copy = None
                allow_tf32_copy = {'tf32_cudnn':False, 'tf32_matmul':False}
                args_copy = to_dtype(args_copy, dtype=torch.float32)
                kwargs_copy = to_dtype(kwargs_copy, dtype=torch.float32)
                grad_out_copy = to_dtype(grad_out_copy, dtype=torch.float32)
                baseline_outputs = self.forward(
                    op, args_copy, kwargs_copy, autocast_config_copy, grad_enable, allow_tf32_copy)
            else:
                raise e
        baseline_grad_inputs = self.backward(baseline_outputs, grad_out_copy, args_copy, kwargs_copy)
        return baseline_outputs, baseline_grad_inputs


    def run(self, device, cpu_baseline=False, configs={}, overflow_check=False):
        op, args, kwargs, autocast_config, grad_enable, allow_tf32 = self.get_op_and_inputs(device)
        if op is None:
            raise ValueError("op is None for {}".format(self.info()))

        grad_out = self.get_grad_out(map_location=device)
        args = replace_device_arg(args, device)
        kwargs = replace_device_arg(kwargs, device)
        if autocast_config is not None:
            autocast_config["device_type"] = device

        if (cpu_baseline and is_cpu_not_supported(self.api)):
            logger.warning("skip {}".format(self.info()))
            return None, None, None, None
        logger.info("run {} ".format(self.info()))

        baseline_outputs = None
        baseline_grad_inputs = None
        if configs.get("dynamic_accuracy_check", None):
            # replay fp64 cuda/cpu data
            baseline_outputs, baseline_grad_inputs = self.run_offline_baseline(
                op,
                args,
                kwargs,
                grad_out,
                autocast_config,
                grad_enable,
                allow_tf32,
                baseline_device=device,
                dynamic_check=True)
            logger.debug("baseline outputs : {}".format(baseline_outputs))
            logger.debug("baseline grad inputs : {}".format(baseline_grad_inputs))
            # dumped mlu data
            mlu_outputs = self.get_outputs(map_location="cpu")
            logger.debug("dumped outputs: {}".format(mlu_outputs))
            mlu_grad_inputs = self.get_grad_inputs(map_location="cpu")
            logger.debug("dumped grad inputs: {}".format(mlu_grad_inputs))
            # replay fp32/fp16/bf16 cuda or cpu data
            outputs = self.forward(op, args, kwargs, autocast_config, grad_enable, allow_tf32)
            logger.debug("replay outputs : {}".format(outputs))
            grad_inputs = self.backward(outputs, grad_out, args, kwargs)
            logger.debug("replay grad inputs : {}".format(grad_inputs))
            return baseline_outputs, baseline_grad_inputs, mlu_outputs, mlu_grad_inputs, outputs, grad_inputs
        elif overflow_check:
            baseline_outputs = self.forward(op, args, kwargs, autocast_config, grad_enable, allow_tf32)
            baseline_grad_inputs = self.backward(baseline_outputs, grad_out, args, kwargs)
            if device == "cpu" and HAS_MLU and torch.mlu.is_available():
                mlu_outputs, mlu_grad_inputs = self.run_offline_baseline(
                    op,
                    args,
                    kwargs,
                    grad_out,
                    autocast_config,
                    grad_enable,
                    allow_tf32,
                    baseline_device="mlu",
                    dynamic_check=False)
            else:
                mlu_outputs = self.get_outputs(map_location="cpu")
                mlu_grad_inputs = self.get_grad_inputs(map_location="cpu")
                if mlu_grad_inputs is not None and baseline_grad_inputs is None:
                    logger.warning("for {}, baseline_grad_inputs is None but mlu_grad_inputs is not None, please check.".format(self.info()))
                    mlu_grad_inputs = None
            # ops of CPU, CUDA and MLU may be implemented differently.
            if not isinstance(get_grad_fn(mlu_outputs), type(get_grad_fn(baseline_outputs))):
                logger.warning("grad_fn in {} is different with baseline device. compare maybe failed.".format(device))
            return mlu_outputs, baseline_outputs, mlu_grad_inputs, baseline_grad_inputs
        else:
            if cpu_baseline:
                # run cpu as baseline
                baseline_outputs, baseline_grad_inputs = self.run_offline_baseline(
                    op,
                    args,
                    kwargs,
                    grad_out,
                    autocast_config,
                    grad_enable,
                    allow_tf32,
                    baseline_device='cpu',
                    dynamic_check=False)
            else:
                # dumped data as baseline
                baseline_outputs = self.get_outputs(map_location="cpu")
                baseline_grad_inputs = self.get_grad_inputs(map_location="cpu")

            logger.debug("baseline outputs : {}".format(baseline_outputs))
            logger.debug("baseline grad inputs : {}".format(baseline_grad_inputs))
            outputs = self.forward(op, args, kwargs, autocast_config, grad_enable, allow_tf32)
            logger.debug("replay outputs : {}".format(outputs))
            # ops of CPU, CUDA and MLU may be implemented differently.
            if cpu_baseline and not isinstance(get_grad_fn(outputs), type(get_grad_fn(baseline_outputs))):
                logger.warning("grad_fn in {} is different with baseline device. compare maybe failed.".format(device))

            grad_inputs = self.backward(outputs, grad_out, args, kwargs)
            logger.debug("replay grad inputs : {}".format(grad_inputs))

            if grad_inputs is None and baseline_grad_inputs is not None:
                logger.warning("for {}, grad_inputs is None but baseline_grad_inputs is not None, please check.".format(self.info()))
                baseline_grad_inputs = None
            return outputs, baseline_outputs, grad_inputs, baseline_grad_inputs
