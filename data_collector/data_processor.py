import os
import threading
import torch
import torch.nn as nn
from torchdump.utils import (
    HIGH,
    MIDDLE,
    LOW,
    get_autocast_config,
    get_allow_tf32_config,
    is_structseq,
    check_overflow_tensor,
    get_logger,
    check_if_include_dtensor_or_faketensor,
)
from torchdump.free_benchmark.utils import create_pre_data_item, safe_args_copy
from torchdump.free_benchmark.disturb_factor.creator import create_disturb_factor
from torchdump.free_benchmark.mode_handler.creator import create_mode_handler
from torchdump.free_benchmark.grad_checker import GradChecker


logger = get_logger()

MAX_DEPTH = 10


class BaseDataProcessor:
    def __init__(self, config):
        self.config = config

    def analyze_forward_inputs(self, args, kwargs, op_name, ctx, dump_func, wrapped_depth=None, hook_instance=None, stack=''):
        op_name = op_name + ".fwd"
        with ctx._dump_seqs_lock:
            if ctx.check_dump_input(op_name, wrapped_depth, hook_instance):
                dump_cont = {}
                if ctx.dump_level > LOW:
                    if len(args) > 0 and isinstance(args[0], nn.Module):
                        # "module" key will be deprecated.
                        dump_cont.update({"module": args[0], "args": args, "kwargs": kwargs})
                    elif len(args) > 0 and isinstance(args[0], torch.optim.Optimizer):
                        # "optim" key will be deprecated.
                        dump_cont.update({"optim": args[0], "args": args, "kwargs": kwargs})
                    else:
                        dump_cont.update({"args": args, "kwargs": kwargs})
                    autocast_config = get_autocast_config(ctx._device) or \
                                      get_autocast_config('cpu')
                    dump_cont["autocast_config"] = autocast_config
                    # record whether grad mode is enabled to save time and avoid
                    # leaf requires grad variable inplace error when replay
                    dump_cont["grad_enable"] = torch.is_grad_enabled()
                    dump_cont["allow_tf32"] = get_allow_tf32_config(ctx._device)
                if stack:
                    dump_cont["stack"] = stack
                dump_func(ctx, dump_cont, op_name, True, hook_instance, True)
        return False

    def analyze_forward_outputs(self,
                                res,
                                op_name,
                                ctx,
                                dump_func,
                                wrapped_depth=None,
                                hook_instance=None,
                                stack='',
                                has_forward_inputs_overflow=False):
        op_name = op_name + ".fwd"
        with ctx._dump_seqs_lock:
            if ctx.check_dump_output(op_name, wrapped_depth, hook_instance):
                dump_cont = {}
                if ctx.dump_level > LOW:
                    dump_res = res
                    if is_structseq(res):
                        dump_res = tuple(res)
                    dump_cont["res"] = dump_res
                if stack:
                    dump_cont["stack"] = stack
                dump_func(ctx, dump_cont, op_name, False, hook_instance)
        return res

    def analyze_backward_inputs(self, grad_out, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        op_name = op_name + ".bwd"
        with ctx._dump_seqs_lock:
            if ctx.check_dump_input(op_name, wrapped_depth):
                dump_cont = {}
                if ctx.dump_level > LOW:
                    dump_cont.update({"grad_out": grad_out})
                if stack:
                    dump_cont["stack"] = stack
                dump_func(ctx, dump_cont, op_name, True, hook_instance)

    def analyze_backward_outputs(self, grad_in, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        op_name = op_name + ".bwd"
        with ctx._dump_seqs_lock:
            if ctx.check_dump_output(op_name, wrapped_depth):
                dump_cont = {}
                if ctx.dump_level > LOW:
                    dump_cont.update({"grad_in": grad_in})
                if stack:
                    dump_cont["stack"] = stack
                dump_func(ctx, dump_cont, op_name, False, hook_instance)


class DefaultDataProcessor(BaseDataProcessor):
    pass


class OverflowCheckDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self._real_overflow_nums, self._lock = 0, threading.RLock()
        self.overflow_nums = self.config.overflow_nums
        self._tls_data = threading.local()

    @property
    def has_overflow(self):
        '''has_overflow need be threading local to ensure correct dump logic.
        When one operator has overflow in current thread, the operator in
        another thread maybe has no overflow. So we cannot make it shared.'''
        return getattr(self._tls_data, "has_overflow", False)

    @has_overflow.setter
    def has_overflow(self, value):
        self._tls_data.has_overflow = value

    @property
    def real_overflow_nums(self):
        with self._lock:
            return self._real_overflow_nums

    def try_add_real_overflow_nums(self):
        with self._lock:
            if self.overflow_nums != -1 and self._real_overflow_nums >= self.overflow_nums:
                return False
            self._real_overflow_nums += 1
            return True

    @property
    def is_terminated(self):
        if self.overflow_nums == -1:
            return False
        if self.real_overflow_nums >= self.overflow_nums:
            return True
        return False

    @classmethod
    def recursive_apply_transform(cls, args, transform, depth=0):
        if depth > MAX_DEPTH:
            raise Exception(f"The maximum depth of recursive transform, {MAX_DEPTH} is reached.")
        if isinstance(args, torch.Tensor):
            tensor_stat = transform(args)
            return [tensor_stat]
        elif isinstance(args, (list, tuple)):
            tensor_stat_list = []
            for i, arg in enumerate(args):
                tensor_stat_list += cls.recursive_apply_transform(arg, transform, depth=depth + 1)
            return tensor_stat_list
        elif isinstance(args, dict):
            tensor_stat_list = []
            for k, arg in args.items():
                tensor_stat_list += cls.recursive_apply_transform(arg, transform, depth=depth + 1)
            return tensor_stat_list
        elif args is not None:
            logger.debug(f"Data type {type(args)} is not supported.")
            return [None]
        else:
            return [None]

    def analyze_single_element(self, data):
        tensor_stat = False
        if not isinstance(data, torch.Tensor):
            logger.warning(f"only handle torch.Tensor, but current data is {type(data)}")
        else:
            tensor_stat = check_overflow_tensor(data)
        return tensor_stat

    def analyze_element(self, element):
        if check_if_include_dtensor_or_faketensor(element):
            return [None]
        return self.recursive_apply_transform(element, self.analyze_single_element)

    def handle_overflow(self, tensor_overflow_list):
        for overflow_stat in tensor_overflow_list:
            if overflow_stat is None:
                continue
            if overflow_stat:
                self.has_overflow = True
                break

    def remove_input_dumped_file_if_existed(self, ctx, op_name):
        dump_seqs_path = ctx.dump_seqs_path
        input_dump_path = os.path.join(os.path.dirname(dump_seqs_path), op_name + ".fwd.input.pt")
        if os.path.exists(input_dump_path):
            os.remove(input_dump_path)
            # remove_dump_file_last_item maybe incorrect in some multi-thread cases, so
            # we remove specific item by dump file name for dump_seqs.
            ctx.remove_dump_file_for_item(input_dump_path.split('/')[-1], dump_seqs_path)

    def analyze_forward_inputs(self, args, kwargs, op_name, ctx, dump_func, wrapped_depth=None, hook_instance=None, stack=''):
        if self.is_terminated:
            return False
        self.has_overflow = False
        forward_tensor_stat_list = self.analyze_element(args)
        forward_tensor_stat_list += self.analyze_element(kwargs)
        self.handle_overflow(forward_tensor_stat_list)
        has_forward_inputs_overflow = self.has_overflow
        super().analyze_forward_inputs(args, kwargs, op_name, ctx, dump_func, wrapped_depth, hook_instance, stack)
        return has_forward_inputs_overflow

    def analyze_forward_outputs(self,
                                outputs,
                                op_name,
                                ctx,
                                dump_func,
                                wrapped_depth=None,
                                hook_instance=None,
                                stack='',
                                has_forward_inputs_overflow=False):
        if self.is_terminated:
            # Input file maybe dumped even if is_terminated is True in multi-thread scenario
            self.remove_input_dumped_file_if_existed(ctx, op_name)
            return outputs
        _wrapped_depth = wrapped_depth if wrapped_depth is not None else ctx.wrapped_depth
        self.has_overflow = False
        forward_tensor_stat_list = self.analyze_element(outputs)
        self.handle_overflow(forward_tensor_stat_list)
        has_forward_outputs_overflow = self.has_overflow
        if _wrapped_depth == 1:
            # Check and update overflow count atomically because multiple threads
            # can overflow at the same time when overflow_nums is small.
            if (has_forward_inputs_overflow or has_forward_outputs_overflow) and self.try_add_real_overflow_nums():
                outputs = super().analyze_forward_outputs(outputs, op_name, ctx, dump_func, _wrapped_depth, hook_instance, stack)
            else:
                self.remove_input_dumped_file_if_existed(ctx, op_name)
        return outputs


class FreeBenchmarkDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        super().__init__(config)
        # cache inputs before op's forward(), and use them after forward().
        # To avoid that inplace ops change inputs after forward().
        self.args = None
        self.kwargs = None
        # flag for skip DTensor
        self.dtensor_flag = False

    def analyze_forward_inputs(self, args, kwargs, op_name, ctx, dump_func, wrapped_depth=None, hook_instance=None, stack=''):
        try:
            if self.config.stage in ["forward", "all"]:
                fwd_op_name = op_name + ".fwd"
                if ctx.check_dump_input(fwd_op_name, wrapped_depth):
                    if check_if_include_dtensor_or_faketensor([args, kwargs]):
                        self.dtensor_flag = True
                        logger.warning(f"FreeBenchmark task does not support DTensor/FakeTensor now, skip for op: {op_name}")
                        return False
                    # if enable verify mode for all stage, we need keep inputs in autograd graph.
                    if self.config.mode == "verify" and self.config.stage == "all":
                        self.args = safe_args_copy(args, keep_in_autograd=True)
                        self.kwargs = safe_args_copy(kwargs, keep_in_autograd=True)
                    else:
                        self.args = safe_args_copy(args, keep_in_autograd=False)
                        self.kwargs = safe_args_copy(kwargs, keep_in_autograd=False)
            # only "backward/all + check" need create GradChecker
            if self.config.stage in ["backward", "all"] and self.config.mode == "check":
                bwd_op_name = op_name + ".bwd"
                if ctx.check_dump_input(bwd_op_name, wrapped_depth):
                    assert hook_instance is not None
                    orig_op = hook_instance._op
                    grad_checker = GradChecker(bwd_op_name, orig_op, self.config.disturb_factor, ctx.cur_iter, ctx.output_dir)
                    grad_checker.saved_kwargs = safe_args_copy(kwargs, keep_in_autograd=False)
                    grad_checker.register_tensor_grad_hook(args)
                    grad_checker.save_forward_inputs_for_vjp(args)
                    setattr(hook_instance, "grad_checker", grad_checker)
        except Exception as e:
            logger.warning(f"[Free Benchmark] cache forward inputs failed for {op_name}: {e}")

        return False

    def analyze_forward_outputs(self,
                                res,
                                op_name,
                                ctx,
                                dump_func,
                                wrapped_depth=None,
                                hook_instance=None,
                                stack='',
                                has_forward_inputs_overflow=False):
        if self.dtensor_flag:
            return res
        if self.config.stage in ["forward", "all"]:
            fwd_op_name = op_name + ".fwd"
            if ctx.check_dump_output(fwd_op_name, wrapped_depth):
                try:
                    if check_if_include_dtensor_or_faketensor(res):
                        self.dtensor_flag = True
                        logger.warning(f"FreeBenchmark task does not support DTensor/FakeTensor now, skip for op: {op_name}")
                        return res
                    assert hook_instance is not None
                    orig_op = hook_instance._op
                    data_item = create_pre_data_item(orig_op, self.args, self.kwargs)
                    data_item.origin_res = res
                    data_item.stage = self.config.stage

                    # create and run disturbed factor
                    factor = create_disturb_factor(self.config.disturb_factor, fwd_op_name)
                    factor.run(data_item)

                    # create and run mode handler: check or verify
                    handler = create_mode_handler(self.config.mode, fwd_op_name, self.config.stage, ctx.output_dir)
                    disturb_res = handler.handle(data_item, self.config.disturb_factor, ctx.cur_iter)

                    # update output for 'verify' mode
                    if handler.need_replace_output():
                        # TODO(): Currently, throw warnings for inplace ops in verify mode
                        orig_name = op_name.rsplit('.', maxsplit=1)[0]
                        if (orig_name.endswith('_') and not orig_name.endswith("__")) \
                                or self.kwargs.get('out', None) is not None \
                                or self.kwargs.get('inplace', False):
                            logger.warning(f"[Free Benchmark] Current op {op_name} is an inplace op, "
                                           "the input may be not replaced completely in verify mode.")
                        logger.debug(f"[Free Benchmark] {op_name}'s forward output will be replaced by disturbed output!")
                        return disturb_res
                except Exception as e:
                    logger.warning(f"[Free Benchmark] run forward disturb and handler failed for {op_name}: {e}")
        return res

    def analyze_backward_inputs(self, grad_out, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        if self.dtensor_flag:
            return
        # only "backward/all + check" need GradChecker
        if self.config.stage in ["backward", "all"] and self.config.mode == "check":
            bwd_op_name = op_name + ".bwd"
            if ctx.check_dump_input(bwd_op_name, wrapped_depth):
                try:
                    if check_if_include_dtensor_or_faketensor(grad_out):
                        logger.warning(f"FreeBenchmark task does not support DTensor/FakeTensor now, skip for op: {op_name}")
                        return
                    grad_checker = getattr(hook_instance, "grad_checker")
                    grad_checker.generate_grad_input_from_vjp(grad_out)
                except Exception as e:
                    logger.warning(f"[Free Benchmark] generate grad_input from vjp failed for {op_name}: {e}")

    def analyze_backward_outputs(self, grad_in, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        # FreeBenchmark only needs grad_out to construct VJP inputs for GradChecker.
        # It does not persist or compare backward outputs through DataProcessor.
        return
