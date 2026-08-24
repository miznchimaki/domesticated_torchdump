import functools
import torch

from contextlib import contextmanager

from torchdump.free_benchmark.mode_handler.creator import create_mode_handler
from torchdump.free_benchmark.disturb_factor.creator import create_disturb_factor
from torchdump.free_benchmark.utils import DataItem, safe_args_copy
from torchdump.utils import get_logger
from torchdump.free_benchmark.utils import create_pre_data_item


logger = get_logger()


_IN_GRAD_CHECKER = False


def is_in_grad_checker():
    return _IN_GRAD_CHECKER

@contextmanager
def flag_context():
    global _IN_GRAD_CHECKER
    old_flag = _IN_GRAD_CHECKER
    _IN_GRAD_CHECKER = True
    try:
        yield
    finally:
        _IN_GRAD_CHECKER = old_flag

def flag_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with flag_context():
            return func(*args, **kwargs)

    return wrapper


class GradChecker:
    def __init__(self, op_name, orig_op, disturb_factor, step, output_dir) -> None:
        self.op_name = op_name
        self.orig_op = orig_op
        self.disturb_factor = disturb_factor
        self.step = step
        self.output_dir = output_dir
        self.origin_grad_input = None
        self.disturb_grad_input = None
        self.saved_inputs = None
        self.saved_kwargs = None
        self.skip_check = False

    def register_tensor_grad_hook(self, inputs):
        hook_idx = 0

        @flag_decorator
        def check_grad_hook(new_idx, grad):
            if self.skip_check:
                return
            try:
                logger.debug(f"[Free Benchmark] START run grad hook for input tensor, current op: {self.op_name}")
                if isinstance(self.disturb_grad_input, str):
                    logger.debug(f"[Free Benchmark] no disturb is added, skip checking grad for current op: {self.op_name}")
                    return
                if self.disturb_grad_input is None:
                    logger.debug(f"[Free Benchmark] disturb_grad_input does not exist, skip checking grad for current op: {self.op_name}")
                    return
                if len(self.disturb_grad_input) <= new_idx:
                    raise Exception(f"Index {new_idx} is out of range for disturb_grad_input of {self.op_name}")
                handler = create_mode_handler("check", self.op_name, "backward", self.output_dir)
                data_item = DataItem(
                                origin_res=grad,
                                disturb_res=self.disturb_grad_input[new_idx].clone(),
                            )
                logger.debug(f"[Free Benchmark] run first grad checker for {new_idx}th grad_input of {self.op_name}...\n" + \
                            f"\torigin_grad_input from network: {data_item.origin_res}\n" + \
                            f"\tdisturb_grad_input from vjp: {data_item.disturb_res}")
                handler.handle(data_item, self.disturb_factor, self.step, error_save=False)
                # Here, need to run a second grad checker if the first check is failed.
                # Because the first check maybe misreported, when current tensor (that the
                # hook is registered on) is the input of multiple ops, its grad (the
                # origin_res we used) is accumulated by calculated grads of all these ops.
                # In such case, the origin grad will not be consitent with disturb grad
                # we calculated from vjp, because vjp only uses one of these ops.
                # In the second checker, both origin_grad and disturb_grad we used for
                # compare are calculated from vjp.
                if handler.is_check_failed():
                    self.skip_check = True
                    handler.reset_check_failed()
                    data_item.origin_res = self.origin_grad_input
                    data_item.disturb_res = self.disturb_grad_input
                    logger.debug(f"[Free Benchmark] run double grad checker for {self.op_name}...\n" + \
                                f"\torigin_grad_input from vjp: {data_item.origin_res}\n" + \
                                f"\tdisturb_grad_input from vjp: {data_item.disturb_res}")
                    handler.handle(data_item, self.disturb_factor, self.step, error_save=True)
            except Exception as e:
                logger.warning(f"[Free Benchmark] run compare for grad_input failed for {self.op_name}: {e}")
            logger.debug(f"[Free Benchmark] END run grad hook for input tensor, current op: {self.op_name}")

        for obj in inputs:
            if isinstance(obj, (tuple, list)):
                for inner_obj in obj:
                    if isinstance(inner_obj, torch.Tensor) and inner_obj.requires_grad:
                        inner_obj.register_hook(functools.partial(check_grad_hook, hook_idx))
                        hook_idx += 1
            elif isinstance(obj, torch.Tensor) and obj.requires_grad:
                obj.register_hook(functools.partial(check_grad_hook, hook_idx))
                hook_idx += 1

    def save_forward_inputs_for_vjp(self, inputs):
        self.saved_inputs = []
        for obj in inputs:
            if isinstance(obj, (tuple, list)):
                item_tuple = []
                for o in obj:
                    item_tuple.append(self._save_forward_inputs_for_vjp_single(o))
                self.saved_inputs.append(item_tuple)
            else:
                self.saved_inputs.append(self._save_forward_inputs_for_vjp_single(obj))

    def _save_forward_inputs_for_vjp_single(self, obj):
        with torch.no_grad():
            if isinstance(obj, torch.Tensor):
                return {
                            "saved_device": obj.device,
                            "saved_tensor": obj.detach().cpu(),
                            "saved_requires_grad": obj.requires_grad,
                    }
            else:
                return obj

    def get_inputs_for_vjp(self):
        # vjp only can input a Tensor or a tuple of Tensors
        input_args = []
        input_tensors = []
        if self.saved_inputs is None:
            raise Exception("get_inputs_for_vjp failed: save forward_inputs for vjp first!")
        for obj in self.saved_inputs:
            if isinstance(obj, list):
                arg_lst = []
                tensor_lst = []
                for o in obj:
                    ret_arg, ret_tensor = self._get_inputs_for_vjp_single(o)
                    arg_lst.append(ret_arg)
                    if ret_tensor is not None:
                        tensor_lst.append(ret_tensor)
                input_args.append(arg_lst)
                input_tensors.extend(tensor_lst)
            else:
                ret_arg, ret_tensor = self._get_inputs_for_vjp_single(obj)
                input_args.append(ret_arg)
                if ret_tensor is not None:
                    input_tensors.append(ret_tensor)
        return input_tensors, tuple(input_args)

    def _get_inputs_for_vjp_single(self, obj):
        ret_arg = None
        ret_tensor = None
        if isinstance(obj, dict) and "saved_tensor" in obj.keys():
            arg = obj["saved_tensor"].clone().detach().to(obj["saved_device"]).requires_grad_(obj["saved_requires_grad"])
            if arg.requires_grad:
                ret_tensor = arg
                ret_arg = "place holder"
            else:
                ret_arg = arg
        else:
            ret_arg = obj
        return ret_arg, ret_tensor

    def run_vjp(self, v, input_tensors, input_args):
        def func(*inputs):
            if self.saved_kwargs is None:
                raise Exception("calculate_vjp failed: save forward_kwargs for vjp first!")
            kwargs = safe_args_copy(self.saved_kwargs, keep_in_autograd=False)
            all_input_args = []
            tensor_idx = 0
            for obj in input_args:
                if obj == "place holder":
                    all_input_args.append(inputs[tensor_idx])
                    tensor_idx += 1
                elif isinstance(obj, list):
                    arg_lst = []
                    for o in obj:
                        if o == "place holder":
                            arg_lst.append(inputs[tensor_idx])
                            tensor_idx += 1
                        else:
                            arg_lst.append(o)
                    all_input_args.append(arg_lst)
                else:
                    all_input_args.append(obj)
            return self.orig_op(*all_input_args, **kwargs)

        _, grad_input = torch.autograd.functional.vjp(func, tuple(input_tensors), v)
        return grad_input

    def generate_disturb_grad_input(self, v, input_tensors, input_args):
        data_item = create_pre_data_item(self.run_vjp, [v, input_tensors, input_args], {})
        # create and run disturbed factor
        factor = create_disturb_factor(self.disturb_factor, self.op_name)
        factor.run(data_item)
        # save disturb_grad_input if not None
        if isinstance(data_item.disturb_res, str):
            self.disturb_grad_input = data_item.disturb_res
        elif data_item.disturb_res is not None:
            self.disturb_grad_input = tuple(t.cpu() for t in data_item.disturb_res)

    @flag_decorator
    def generate_grad_input_from_vjp(self, grad_out):
        # calculate and save origin grad_input using vjp
        input_tensors, input_args = self.get_inputs_for_vjp()
        # deepcopy input_args to avoid being changed by inplace op
        input_args_copy = safe_args_copy(input_args, keep_in_autograd=False)
        origin_grad_input = self.run_vjp(grad_out, input_tensors, input_args)
        self.origin_grad_input = tuple(t.cpu() for t in origin_grad_input)

        # calculate and save disturb grad_input using vjp
        # call get_inputs_for_vjp() again to avoid inputs being changed by inplace op
        input_tensors, _ = self.get_inputs_for_vjp()
        self.generate_disturb_grad_input(grad_out, input_tensors, input_args_copy)

