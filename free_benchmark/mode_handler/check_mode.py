import torch
import os
import torch.utils._pytree as pytree
import numpy as np

from filelock import FileLock
from dataclasses import asdict, dataclass

from torchdump.free_benchmark.mode_handler.base import BaseModeHandler
from torchdump.free_benchmark.utils import create_new_csv_row, format_floats_in_dict, write_to_csv
from torchdump.utils import (
    get_logger,
    calc_static_diff,
    check_all_values_pass,
    get_size,
)

torch_version_less_2_2=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.2')
torch_version_less_2_7=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.7')
torch_version_less_2_11=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.11')
torch_version_less_2_12=(lambda v1,v2:next((int(x)<int(y) for x,y in zip(v1.split('.'),v2.split('.'))if x!=y),False))(torch.__version__.split('+')[0],'2.12')

logger = get_logger()

class ThresholdManager:
    # Used for comparison between MLU and CPU
    CPU_STATIC_THD_CONFIG = {
        "diff1_static_threshold": 3e-3,
        "diff2_static_threshold": 3e-3,
    }

    # Used for comparison between MLU and MLU
    # applied to disturbed output, to clamp the min value
    MLU_ABS_TOL_PER_DTYPE_MAP = {
        torch.bfloat16: 1e-2,
        torch.float16: 1e-3,
        torch.float32: 1e-4,
        torch.float64: 1e-8,
    }
    MLU_BACKWARD_OUTPUT_ABS_TOL = 1e-3
    # applied to original output, to confirm the rtol
    MLU_RATIO_THD_CONFIG = {
        torch.float8_e5m2: 1.05,
        torch.float8_e4m3fn: 1.05,
        torch.bfloat16: 1.004,
        torch.float16: 1.002,
        torch.float32: 1.0002,
    }
    if not torch_version_less_2_2:
        MLU_RATIO_THD_CONFIG[torch.float8_e5m2fnuz] = 1.05
        MLU_RATIO_THD_CONFIG[torch.float8_e4m3fnuz] = 1.05
    if not torch_version_less_2_7:
        MLU_RATIO_THD_CONFIG[torch.float8_e8m0fnu] = 1.05
    if not torch_version_less_2_11:
        MLU_RATIO_THD_CONFIG[torch.float4_e2m1fn_x2] = 1.05
        MLU_ABS_TOL_PER_DTYPE_MAP[torch.float4_e2m1fn_x2] = 1e-1

    @classmethod
    def get_cpu_static_threshold_config(cls,):
        return cls.CPU_STATIC_THD_CONFIG

    @classmethod
    def set_cpu_static_threshold_config(cls, config_dict={}):
        assert isinstance(config_dict, dict), "The type of threshold config for mlu and cpu must be dict."
        allowed_keys = {"diff1_static_threshold", "diff2_static_threshold", "dtype_static_thresholds"}
        assert all(key in allowed_keys for key in config_dict.keys()), \
                "The key of threshold config dict for mlu and cpu must be one of \"diff1_static_threshold\", \"diff2_static_threshold\", or \"dtype_static_thresholds\"."
        if "dtype_static_thresholds" in config_dict:
            assert isinstance(config_dict["dtype_static_thresholds"], dict), \
                    "The value of dtype_static_thresholds must be dict type."
        cls.CPU_STATIC_THD_CONFIG = config_dict

    @classmethod
    def get_mlu_ratio_threshold_config(cls,):
        return cls.MLU_RATIO_THD_CONFIG

    @classmethod
    def set_mlu_ratio_threshold_config(cls, config_dict={}):
        assert isinstance(config_dict, dict), "The type of threshold config for mlu and mlu must be dict."
        assert all(isinstance(key, torch.dtype) for key in config_dict.keys()), \
                "The key of threshold config dict for mlu and mlu must be torch.dtype type."
        assert all(isinstance(val, float) for val in config_dict.values()), \
                "The value of threshold config dict for mlu and mlu must be float type."
        cls.MLU_RATIO_THD_CONFIG = config_dict


class CheckModeHandler(BaseModeHandler):
    def __init__(self, op_name, stage, output_dir) -> None:
        super().__init__(op_name, stage)
        self.err_row = None
        self.output_dir = output_dir
        self._check_failed = False

    def handle(self, data_item, disturb_factor, step, error_save=True):
        if data_item.disturb_res == "NO NEED":
            logger.debug(f"[Free Benchmark] no need to disturb for {self.op_name}, skip check it.")
            return data_item.origin_res
        elif data_item.disturb_res == "FAILED":
            logger.debug(f"[Free Benchmark] failed to disturb for {self.op_name}, skip check it.")
            return data_item.origin_res
        try:
            if disturb_factor == "to_cpu":
                self.check_with_cpu_output(data_item, step)
            else: # for disturb_factor "type_promotion"
                self.check_with_mlu_output(data_item, step)
            if error_save:
                self.append_row_to_result_csv()
        except Exception as e:
            logger.warning(f"[Free Benchmark] check api {self.op_name} failed: {e}")
        return data_item.origin_res

    def is_check_failed(self):
        return self._check_failed

    def reset_check_failed(self):
        self._check_failed = False
        self.err_row = None

    def append_row_to_result_csv(self):
        # update error row to free_benchmark csv
        if self.is_check_failed():
            assert self.err_row is not None, "check failed but no err_row is created!"
            row_dict = asdict(self.err_row)
            os.makedirs(self.output_dir, exist_ok=True)
            csv_path = os.path.join(self.output_dir, "free_benchmark.csv")
            lock_path = csv_path + ".lock"
            with FileLock(lock_path):
                write_to_csv(row_dict.values(), row_dict.keys(), csv_path)
            # empty err_row
            self.err_row = None

    def check_with_cpu_output(self, data_item, step):
        res_dict = calc_static_diff(data_item.origin_res, data_item.disturb_res, configs=ThresholdManager.CPU_STATIC_THD_CONFIG)
        logger.debug(f"[Free Benchmark] {self.op_name} - res_dict of check_with_cpu: {res_dict}")
        if not check_all_values_pass(res_dict):
            self._check_failed = True
            self.err_row = create_new_csv_row(data_item, self.op_name, "to_cpu", step, error_message=str(format_floats_in_dict(res_dict)))

    def check_with_mlu_output(self, data_item, step):
        res_dict = CheckModeHandler.calc_max_ratio_mlu(data_item.origin_res, data_item.disturb_res, self.stage)
        logger.debug(f"[Free Benchmark] {self.op_name} - res_dict of check_with_mlu: {res_dict}")
        if not check_all_values_pass(res_dict):
            self._check_failed = True
            self.err_row = create_new_csv_row(data_item, self.op_name, "type_promotion", step, error_message=str(format_floats_in_dict(res_dict)))

    @staticmethod
    def calc_max_ratio_mlu(origin, disturb, stage):
        if origin is None and disturb is None:
            return {}
        assert isinstance(origin, type(disturb)), \
                f"original output and disturb output must be the same type, got ({type(origin)}, {type(disturb)})."
        assert get_size(origin) == get_size(disturb), \
                f"original output and disturb output must be the same length, got ({get_size(origin)}, {get_size(disturb)})."

        results = {}
        if isinstance(origin, (list, tuple)):
            for i in range(len(origin)):
                res = CheckModeHandler.calc_max_ratio_mlu(origin[i], disturb[i], stage)
                # only save failed element
                if not check_all_values_pass(res):
                    results[i] = res
            # every element of list of tuple are all passed
            if len(results) == 0:
                return {"max_rel": "PASS"}
        elif isinstance(origin, dict):
            # flatten leaf elements to list for compare
            return CheckModeHandler.calc_max_ratio_mlu(pytree.tree_flatten(origin)[0], pytree.tree_flatten(disturb)[0], stage)
        elif isinstance(origin, int):
            if origin == disturb:
                return {"max_rel": "PASS"}
            return {"max_rel": "INT NOT EQUAL"}
        elif isinstance(origin, float):
            return CheckModeHandler.calc_max_ratio_mlu(torch.tensor(origin), torch.tensor(disturb), stage)
        elif isinstance(origin, torch.Tensor):
            if origin.is_meta:
                assert disturb.is_meta == True
                return {"max_rel": "PASS"}
            atol = ThresholdManager.MLU_BACKWARD_OUTPUT_ABS_TOL if stage == "backward" else \
                ThresholdManager.MLU_ABS_TOL_PER_DTYPE_MAP.get(
                    disturb.dtype, ThresholdManager.MLU_ABS_TOL_PER_DTYPE_MAP[torch.float32]
                )
            thr_ratio = ThresholdManager.MLU_RATIO_THD_CONFIG[origin.dtype]
            origin = origin.cpu() if not origin.is_cpu else origin
            disturb = disturb.cpu() if not disturb.is_cpu else disturb
            max_ratio = CheckModeHandler.calc_max_ratio_single(
                origin.detach().to(torch.float64).numpy(), disturb.detach().to(torch.float64).numpy(), atol
            )
            if max_ratio == "INCONSISTENT SIGN":
                results["max_rel"] = "INCONSISTENT SIGN"
            elif max_ratio > thr_ratio:
                results["max_rel"] = max_ratio - 1
        else:
            raise TypeError(f"Unsupported type: {type(origin)}, please add method for this type.")
        return results

    @staticmethod
    def calc_max_ratio_single(origin, disturb, atol):
        # if origin and disturb have inconsistent sign, return flag directly
        if np.min(origin * disturb) < -(atol**2):
            return "INCONSISTENT SIGN"
        # clip min_val to atol
        origin = np.clip(np.abs(origin), a_min=atol, a_max=None)
        disturb = np.clip(np.abs(disturb), a_min=atol, a_max=None)
        # calculate ratio, regard nan as consistent
        ratio1 = np.nan_to_num(origin / disturb, nan=1.0)
        ratio2 = np.nan_to_num(disturb / origin, nan=1.0)
        return max(np.max(ratio1), np.max(ratio2))

