import numpy as np
import warnings

import torch

def calc_diff1(base, eval, epsilon=1e-9):
    assert base.shape == eval.shape, f"input shape must be same, now got {base.shape} and {eval.shape}"
    denominator = np.sum(np.abs(base))
    diff1 = np.sum(np.abs(base - eval)) / np.maximum(denominator, epsilon)
    return diff1


def calc_diff2(base, eval, epsilon=1e-9):
    assert base.shape == eval.shape, f"input shape must be same, now got {base.shape} and {eval.shape}"
    denominator = np.sum(np.square(base))
    diff2 = np.sum(np.square(base - eval)) / np.maximum(denominator, epsilon)
    diff2 = np.sqrt(diff2)
    return diff2


def calc_diff3(base, eval):
    assert base.shape == eval.shape, f"input shape must be same, now got {base.shape} and {eval.shape}"
    diff3 = np.max(np.abs(base - eval))
    return diff3


# diff4 not used now
def calc_gpu_diff4(base, eval):
    assert base.shape == eval.shape, f"input shape must be same, now got {base.shape} and {eval.shape}"
    # if gpu_diff4 == 0 or gpu_diff4 == 1
    # diff4 of mlu will be ignored (set diff4 to -1)
    # in other words, test of diff4 will always be passed
    ne_cnt = np.sum(eval != base).astype(np.float64)
    lt_cnt = np.sum(eval < base).astype(np.float64)
    if ne_cnt == 0:
        # if ne_cnt = 0, set gpu_diff4 to 0.5
        # which means gpu_diff4 will be ignored in the later computation
        gpu_diff4 = 0.5
    else:
        gpu_diff4 = lt_cnt / ne_cnt

    if gpu_diff4 in (0.0, 1.0):
        return -1.0
    else:
        return gpu_diff4


def calc_mlu_diff4(base, eval):
    assert base.shape == eval.shape, f"input shape must be same, now got {base.shape} and {eval.shape}"
    ne_cnt = np.sum(eval != base).astype(np.float64)
    lt_cnt = np.sum(eval < base).astype(np.float64)
    # when mlu_ne_count < 100, diff4 is ignored, it is not a suitable evalation standard
    if ne_cnt < 100:
        mlu_diff4 = -1.0
    else:
        mlu_diff4 = lt_cnt / ne_cnt
    return mlu_diff4


class Evaluator:
    def __init__(self, base, eval_mlu, eval_gpu, configs={}):
        self.base = base.cpu()
        self.eval_mlu = eval_mlu.cpu()
        self.eval_gpu = eval_gpu.cpu()
        self.configs = configs

        self.half_min_threshold = configs.get("half_min_threshold", None)
        self.float_min_threshold = configs.get("float_min_threshold", None)
        if configs.get("standard", None) is None:
            self.configs["standard"] = "default"
        self.max_threshold = 0.0
        self.static_threshold = 0.0
        self.is_floating_diff = False
        self.diff_type = None
        self.init_params()

    def init_params(self):
        if self.eval_mlu.is_complex():
            # set diff type
            if self.eval_mlu.dtype == torch.complex32:
                self.diff_dtype = torch.float16
            elif self.eval_mlu.dtype == torch.complex64:
                self.diff_dtype = torch.float32
            else:
                raise TypeError(
                    "Complex dtype only half and float support dynamic diff"
                )
        else:
            self.diff_dtype = self.eval_mlu.dtype

        # set diff threshold
        if self.diff_dtype in [torch.float16, torch.bfloat16]:
            if self.half_min_threshold:
                self.static_threshold = self.half_min_threshold
            else:
                self.static_threshold = 1e-3
            self.max_threshold = 3e-2
            self.is_floating_diff = True
        elif self.diff_dtype == torch.float32:
            if self.float_min_threshold:
                self.static_threshold = self.float_min_threshold
            else:
                self.static_threshold = 1e-5
            self.max_threshold = 3e-3
            self.is_floating_diff = True
        elif self.diff_dtype == torch.float64:
            warnings.warn(
                "Double is not proper for dynamic diff"
            )
            self.static_threshold = 1e-5
            self.is_floating_diff = True
        else:
            pass

        if self.check_nan_inf(self.eval_mlu) or self.check_nan_inf(self.eval_gpu):
            raise ValueError("Please use static check for nan or inf")

    @torch.no_grad()
    def to_numpy(self, base, eval):
        if self.configs.get("dynamic_threshold", None):
            # eval is fp32/fp16/bf16 data, base is fp64
            eval_np = eval.to(torch.float64).numpy()
            # convert golden fp64 to eval dtype, then convert to fp64
            base_np = base.to(eval.dtype).to(torch.float64).numpy()
            return base_np, eval_np
        elif self.configs.get("static_threshold", None):
            # eval and base have same dtype
            eval_np = eval.to(torch.float64).numpy()
            # convert golden fp64 to eval dtype, then convert to fp64
            base_np = base.to(torch.float64).numpy()
            return base_np, eval_np
        else:
            raise ValueError("dynamic_threshold or static_threshold must be set")

    def calc_diff_base(self, diff_func, base, eval):
        real_diff, imag_diff = -1.0, -1.0
        if eval.is_complex():
            base_real_np, eval_real_np = self.to_numpy(base.real, eval.real)
            base_imag_np, eval_imag_np = self.to_numpy(base.imag, eval.imag)
            real_diff = diff_func(base_real_np, eval_real_np)
            imag_diff = diff_func(base_imag_np, eval_imag_np)
            return real_diff, imag_diff
        else:
            base_np, eval_np = self.to_numpy(base, eval)
            real_diff = diff_func(base_np, eval_np)
            return real_diff

    def calc_dynamic_diff1(self, base, eval):
        if not self.is_floating_diff:
            warnings.warn(
                "\033[1;33m [Evaluator]: diff1 expects float data type, maybe you should use diff3 == 0!\033[0m"
            )
        diff1 = self.calc_diff_base(calc_diff1, base, eval)
        return diff1

    def calc_dynamic_diff2(self, base, eval):
        if not self.is_floating_diff:
            warnings.warn(
                "\033[1;33m [Evaluator]: diff2 expects float data type, maybe you should use diff3 == 0!\033[0m"
            )
        diff2 = self.calc_diff_base(calc_diff2, base, eval)
        return diff2

    def calc_dynamic_diff4(self, device_type="mlu"):
        if device_type == "mlu":
            diff4 = self.calc_diff_base(calc_mlu_diff4, self.base, self.eval_mlu)
        else:
            diff4 = self.calc_diff_base(calc_gpu_diff4, self.base, self.eval_gpu)
        return diff4

    def calc_dynamic_diff(self, base, eval, device_type="mlu"):
        # now we only support standard='default'
        assert self.configs["standard"] in ["default"], f'standard now only support default for dynamic check, not support {self.configs["standard"]}'
        # calc diff1, diff2
        diff1 = self.calc_dynamic_diff1(base, eval)
        diff2 = self.calc_dynamic_diff2(base, eval)
        return diff1, diff2

    def calc_gpu_dynamic_diff(self):
        gpu_diff = self.calc_dynamic_diff(self.base, self.eval_gpu, device_type="cuda")
        return gpu_diff

    def calc_mlu_dynamic_diff(self):
        mlu_diff = self.calc_dynamic_diff(self.base, self.eval_mlu, device_type="mlu")
        return mlu_diff

    def calc_static_diff3(self):
        # for standard='binary' or integer tensors
        diff3 = self.calc_diff_base(calc_diff3, self.eval_gpu, self.eval_mlu)
        return diff3

    def calc_diff(self):
        results = {}
        if torch.is_complex(self.eval_mlu) or torch.is_floating_point(self.eval_mlu):
            if self.configs["standard"] == "binary":
                self.configs["static_threshold"] = True
                diff = self.calc_static_diff3()
                results["diff3"] = diff
                return results
            else:
                # use dynamic diff
                self.configs["dynamic_threshold"] = True
                gpu_diff = self.calc_gpu_dynamic_diff()
                mlu_diff = self.calc_mlu_dynamic_diff()
                results["diff1"] = [gpu_diff[0], mlu_diff[0]]
                results["diff2"] = [gpu_diff[1], mlu_diff[1]]
                return results
        else:
            # for integer tensors
            self.configs["static_threshold"] = True
            diff = self.calc_static_diff3()
            results["diff3"] = diff
            return results

    @torch.no_grad()
    def check_nan_inf(self, tensor):
        return torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()

    def check_dynamic_diff4(self, gpu_diff, mlu_diff):
        # mlu_ne_count < 100 || (mlu_diff4 != 0 && mlu_diff4 != 1 ) || (gpu_diff4 == 0 || gpu_diff4 == 1)
        # mlu_diff < 0 means diff4 is not used because data points are less than 100
        # gpu_diff < 0 means diff4 id not used because gpu_diff4=0.0 or gpu_diff4=1.0
        if mlu_diff >= 0.0 and gpu_diff >= 0.0:
            if mlu_diff == 0.0 or mlu_diff == 1.0:
                return False
        return True

    # used for diff1 and diff2
    def check_dynamic_diff1_and_diff2(self, gpu_diff, mlu_diff, rate=10.0):
        # mlu_diff1: mlu_diff1 <= (max(gpu_diff1, static_thred)) * rate
        # mlu_diff2: mlu_diff2 <= (max(gpu_diff2, static_thred)) * rate
        dynamic_diff = gpu_diff * rate
        min_threshold = self.static_threshold
        if rate > 10.0:
            min_threshold *= rate
        dynamic_diff = np.maximum(dynamic_diff, min_threshold)
        if self.is_floating_diff and rate > 10.0:
            # max_threshold is valid only when rate is over default rate
            dynamic_diff = np.minimum(dynamic_diff, self.max_threshold)

        if mlu_diff <= dynamic_diff:
            return True
        else:
            return False

    def check_static_diff3(self, diff):
        if diff == 0:
            return True
        else:
            return False

    # diffs should be results of calc_diff
    def check_diff(self, diffs):
        results = {}
        if self.configs["standard"] == "binary":
            results["diff3"] = diffs["diff3"]
            if self.eval_mlu.is_complex():
                if self.check_static_diff3(
                    diffs["diff3"][0]
                ) and self.check_static_diff3(diffs["diff3"][1]):
                    results["diff3"] = "PASS"
            else:
                if self.check_static_diff3(diffs["diff3"]):
                    results["diff3"] = "PASS"
        elif self.configs["standard"] == "default":
            if self.eval_mlu.is_floating_point():
                results["diff1"] = diffs["diff1"]
                results["diff2"] = diffs["diff2"]
                if self.check_dynamic_diff1_and_diff2(
                    results["diff1"][0], results["diff1"][1]
                ):
                    results["diff1"] = "PASS"
                if self.check_dynamic_diff1_and_diff2(
                    results["diff2"][0], results["diff2"][1]
                ):
                    results["diff2"] = "PASS"
            elif self.eval_mlu.is_complex():
                results["diff1"] = diffs["diff1"]
                results["diff2"] = diffs["diff2"]
                if self.check_dynamic_diff1_and_diff2(
                    results["diff1"][0][0], results["diff1"][1][0]
                ) and self.check_dynamic_diff1_and_diff2(
                    results["diff1"][0][1], results["diff1"][1][1]
                ):
                    results["diff1"] = "PASS"

                if self.check_dynamic_diff1_and_diff2(
                    results["diff2"][0][0], results["diff2"][1][0]
                ) and self.check_dynamic_diff1_and_diff2(
                    results["diff2"][0][1], results["diff2"][1][1]
                ):
                    results["diff2"] = "PASS"
            else:
                # interger tensor
                results["diff3"] = diffs["diff3"]
                if self.check_static_diff3(results["diff3"]):
                    results["diff3"] = "PASS"
        else:
            raise ValueError("unsupported api standard")
        return results

