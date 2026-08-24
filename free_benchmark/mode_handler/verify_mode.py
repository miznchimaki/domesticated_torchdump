import torch

from numbers import Number

from torchdump.free_benchmark.mode_handler.base import BaseModeHandler
from torchdump.utils import get_logger


logger = get_logger()


class VerifyModeHandler(BaseModeHandler):
    def __init__(self, op_name, stage) -> None:
        super().__init__(op_name, stage)
        self._need_replace_output = False

    def handle(self, data_item, disturb_factor, step, error_save=True):
        if data_item.disturb_res == "NO NEED":
            logger.debug(f"[Free Benchmark] no need to disturb for {self.op_name}, skip verify it.")
            self._need_replace_output = False
            return data_item.origin_res
        elif data_item.disturb_res == "FAILED":
            logger.debug(f"[Free Benchmark] failed to disturb for {self.op_name}, skip verify it.")
            self._need_replace_output = False
            return data_item.origin_res
        try:
            res = VerifyModeHandler.recursive_replace_origin_output(data_item.origin_res, data_item.disturb_res, self.stage)
            self._need_replace_output = True
            return res
        except Exception as e:
            logger.warning(f"[Free Benchmark] verify api {self.op_name} failed: {e}")
            self._need_replace_output = False
            return data_item.origin_res

    def need_replace_output(self):
        return self._need_replace_output

    @staticmethod
    def recursive_replace_origin_output(orig, new, stage):
        if orig is None and new is None:
            return None
        elif isinstance(orig, Number) and isinstance(new, Number):
            return new
        elif isinstance(orig, torch.Tensor) and isinstance(new, torch.Tensor):
            if stage == "forward":
                orig.data = new.to(orig.dtype).to(orig.device)
            else:
                new = new.to(orig.dtype).to(orig.device)
                orig = new
            return orig
        elif isinstance(orig, dict) and isinstance(new, dict):
            res = {}
            for key, value in orig.items():
                if key not in new:
                    raise Exception(f"key '{key}' not found in disturbed output when replacing origin output")
                res[key] = VerifyModeHandler.recursive_replace_origin_output(value, new[key], stage)
            return res
        elif isinstance(orig, (tuple, list)) and isinstance(new, (tuple, list)):
            if len(orig) != len(new):
                raise Exception(f"length {len(orig)} not match length {len(new)} when replacing original output")
            return type(orig)(
                    VerifyModeHandler.recursive_replace_origin_output(value, new[idx], stage) for idx, value in enumerate(orig))
        else:
            raise Exception(f"not support to replace origin type '{type(orig)}' with disturbed type '{type(new)}'")
