from torchdump.utils import get_logger

from .data_factor import  DataProcessorFactory


logger = get_logger()


class DataCollector:
    task_types = ["default", "overflow_check", "free_benchmark"]
    def __init__(self, config):
        self.config = config
        assert self.config.task in self.task_types, f"current task \"{self.config.task}\" not support DataCollector."
        self.data_processor = DataProcessorFactory.create_processor(self.config)

    def need_register_bwd_hook(self):
        if self.config.task == "overflow_check":
            return False
        elif self.config.task == "free_benchmark" and self.config.stage == "forward":
            return False
        return True

    def forward_inputs_data_collect(self, args, kwargs, op_name, ctx, dump_func, wrapped_depth=None, hook_instance=None, stack=''):
        return self.data_processor.analyze_forward_inputs(args, kwargs, op_name, ctx, dump_func, wrapped_depth, hook_instance, stack)

    def forward_outputs_data_collect(self, outputs, op_name, ctx, dump_func, wrapped_depth=None, hook_instance=None, stack='',
                                     has_forward_inputs_overflow=False):
        return self.data_processor.analyze_forward_outputs(outputs, op_name, ctx, dump_func, wrapped_depth, hook_instance, stack,
                                                    has_forward_inputs_overflow)

    def backward_inputs_data_collect(self, grad_out, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        self.data_processor.analyze_backward_inputs(grad_out, op_name, wrapped_depth, ctx, dump_func, hook_instance, stack)

    def backward_outputs_data_collect(self, grad_in, op_name, wrapped_depth, ctx, dump_func, hook_instance=None, stack=''):
        self.data_processor.analyze_backward_outputs(grad_in, op_name, wrapped_depth, ctx, dump_func, hook_instance, stack)
