import torch
import os
import hashlib
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
import torch.distributed as dist

from torchdump.utils import get_logger

logger = get_logger()

CsvHeaderInput = namedtuple("CsvHeaderInput", ["bounds"])
CsvContentInput = namedtuple("CsvContentInput", ["grad", "bounds"])

level_interp = {
    "L0": {
        "header": ["md5", "max", "min", "norm", "shape"],
        "have_grad_direction": False
    },
    "L1": {
        "header": ["max", "min", "norm", "shape"],
        "have_grad_direction": True
    },
    "L2": {
        "header": ["distribution", "max", "min", "norm", "shape"],
        "have_grad_direction": True
    }
}

def check_numeral_increase_list(num_list):
    if any(not isinstance(item, (int, float)) for item in num_list):
        raise Exception("The input list should only contain numbers")
    if num_list != sorted(num_list):
        raise Exception("The input list should be ascending")
    
class GradStatCsv:
    csv = {}
    @staticmethod
    def generate_csv_header(level, bounds):
        header = ["param_name"]
        for key in level["header"]:
            csv_header_input = CsvHeaderInput(bounds=bounds)
            header.extend(GradStatCsv.csv[key].generate_csv_header(csv_header_input))
        return header

    @staticmethod
    def generate_csv_line(param_name, level, grad, bounds):
        line = [param_name]
        for key in level["header"]:
            csv_content_input = CsvContentInput(grad=grad, bounds=bounds)
            line.extend(GradStatCsv.csv[key].generate_csv_content(csv_content_input))
        return line

def register_csv_handler(key, cls=None):
    if cls is None:
        # 无参数时，返回装饰器函数
        return lambda cls: register_csv_handler(key, cls)
    GradStatCsv.csv[key] = cls
    return cls

class CsvHandler(ABC):
    @staticmethod
    @abstractmethod
    def generate_csv_header(csv_header_input):
        pass

    @staticmethod
    @abstractmethod
    def generate_csv_content(csv_content_input):
        pass


@register_csv_handler("distribution")
class CsvDistribution(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        bounds = csv_header_input.bounds
        intervals = []
        if bounds:
            intervals.append(f"(-inf, {bounds[0]}]")
            for i in range(1, len(bounds)):
                intervals.append(f"({bounds[i-1]}, {bounds[i]}]")
        if intervals:
            intervals.append(f"({bounds[-1]}, inf)")
        intervals.append("=0")
        return intervals

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        bounds = csv_content_input.bounds
        grad = grad.cpu().detach()
        element_num = grad.numel()
        grad_equal_0_num = (grad == 0).sum().item()
        bound = torch.Tensor(bounds)
        bucketsize_result = torch.bucketize(grad, bound)
        interval_nums = [(bucketsize_result == i).sum().item() for i in range(len(bound) + 1)]
        interval_nums.append(grad_equal_0_num)
        return_list = [x / element_num if element_num != 0 else 0 for x in interval_nums]
        return return_list

@register_csv_handler("max")
class CsvMax(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["max"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.max(grad).cpu().detach().float().numpy().tolist()]

@register_csv_handler("min")
class CsvMin(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["min"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.min(grad).cpu().detach().float().numpy().tolist()]

@register_csv_handler("norm")
class CsvNorm(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["norm"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [torch.norm(grad).cpu().detach().float().numpy().tolist()]

@register_csv_handler("shape")
class CsvShape(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["shape"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        return [list(grad.shape)]

@register_csv_handler("md5")
class CsvMd5(CsvHandler):
    @staticmethod
    def generate_csv_header(csv_header_input):
        return ["md5"]

    @staticmethod
    def generate_csv_content(csv_content_input):
        grad = csv_content_input.grad
        tensor_bytes = grad.cpu().detach().float().numpy().tobytes()
        md5_hash = hashlib.md5(tensor_bytes)
        return [md5_hash.hexdigest()]

def save_npy(data, filepath):
    filepath = os.path.realpath(filepath)
    try:
        np.save(filepath, data)
    except Exception as e:
        logger.error(f"The numpy file failed to save. Please check the path: {filepath}.")
        raise RuntimeError(f"Save numpy file {filepath} failed.") from e

def print_rank0_message(message, cur_rank):
    if dist.is_initialized():
        if cur_rank == 0:
            logger.info(f"{message}")
    else:
        logger.info(f"{message}")

def is_data_in_list(data, lst):
    return not lst or len(lst) == 0 or data in lst
