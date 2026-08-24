import torch

from torchdump.free_benchmark.disturb_factor.base import BaseFactor
from torchdump.free_benchmark.utils import is_reduced_floating_point
from torchdump.utils import get_logger, to_dtype


logger = get_logger()


class TypePromotionFactor(BaseFactor):
    def __init__(self, op_name):
        super().__init__(op_name)
        self.need_disturb = False

    def run(self, data_item):
        logger.info(f"[Free Benchmark] start to run 'type_promotion' disturb_factor for api: {self.op_name} ...")

        try:
            args_promote = to_dtype(data_item.args, torch.float32, self.disturb_to_dtype_tensor_hanler)
            kwargs_promote = to_dtype(data_item.kwargs, torch.float32, self.disturb_to_dtype_tensor_hanler)
            if self.need_disturb:
                data_item.disturb_res = data_item.origin_op(*args_promote, **kwargs_promote)
            else:
                data_item.disturb_res = "NO NEED"
        except Exception as e:
            logger.warning(f"[Free Benchmark] failed to run 'type_promotion' disturb for {self.op_name}: {e}")
            data_item.disturb_res = "FAILED"

        return data_item.disturb_res

    def disturb_to_dtype_tensor_hanler(self, obj, dtype):
        if is_reduced_floating_point(obj):
            self.need_disturb = True
            return obj.to(dtype)
        return obj
