import torch

from torchdump.free_benchmark.disturb_factor.base import BaseFactor
from torchdump.free_benchmark.utils import is_reduced_floating_point
from torchdump.utils import get_logger, replace_device_arg, to_dtype


logger = get_logger()


class ToCpuFactor(BaseFactor):
    def __init__(self, op_name):
        super().__init__(op_name)
        self.need_disturb = False

    def run(self, data_item):
        logger.info(f"[Free Benchmark] start to run 'to_cpu' disturb_factor for api: {self.op_name} ...")

        try:
            args_cpu = self.recursive_clone_to_cpu(data_item.args)
            kwargs_cpu = self.recursive_clone_to_cpu(data_item.kwargs)
            if self.need_disturb:
                args_cpu = replace_device_arg(args_cpu, device="cpu")
                kwargs_cpu = replace_device_arg(kwargs_cpu, device="cpu")
                try:
                    data_item.disturb_res = data_item.origin_op(*args_cpu, **kwargs_cpu)
                # raise "'op' not implemented for 'dtype'", then cast to float32 and rerun
                except RuntimeError as e:
                    if "not implemented for" in str(e):
                        logger.warning(f"[Free Benchmark] {e}, will cast inputs to FP32 for {self.op_name} and rerun.")
                        args_cpu = to_dtype(args_cpu, torch.float32, ToCpuFactor.promote_to_dtype_tensor_hanler)
                        kwargs_cpu = to_dtype(kwargs_cpu, torch.float32, ToCpuFactor.promote_to_dtype_tensor_hanler)
                        data_item.disturb_res = data_item.origin_op(*args_cpu, **kwargs_cpu)
                    else:
                        raise e
            else:
                data_item.disturb_res = "NO NEED"
        except Exception as e:
            logger.warning(f"[Free Benchmark] failed to run 'to_cpu' disturb for {self.op_name}: {e}")
            data_item.disturb_res = "FAILED"

        return data_item.disturb_res

    def recursive_clone_to_cpu(self, tensor_seq):
        if isinstance(tensor_seq, torch.Tensor):
            if not tensor_seq.is_cpu:
                self.need_disturb = True
            return tensor_seq.to("cpu")
        elif isinstance(tensor_seq, dict):
            ret = {}
            for key, value in tensor_seq.items():
                ret[key] = self.recursive_clone_to_cpu(value)
            return ret
        elif isinstance(tensor_seq, (list, tuple)):
            return type(tensor_seq)(
                    self.recursive_clone_to_cpu(value) for value in tensor_seq)
        else:
            return tensor_seq

    @staticmethod
    def promote_to_dtype_tensor_hanler(obj, dtype):
        # for FP8/FP16/BF16, just convert to FP32 when inputs not supported for cpu op
        if is_reduced_floating_point(obj):
            return obj.to(dtype)
        return obj
