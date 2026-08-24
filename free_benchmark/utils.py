import os
import csv
import copy
import torch

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Callable, Any


@dataclass
class DataItem:
    args: Optional[Tuple] = None
    kwargs: Optional[Dict] = None
    origin_op: Optional[Callable] = None
    origin_res: Optional[Any] = None
    disturb_res: Optional[Any] = None
    stage: Optional[str] = None


@dataclass
class FreeBenchmarkCsvRow:
    rank: Optional[int] = None
    step: Optional[int] = None
    api: Optional[str] = None
    disturb_factor: Optional[str] = None
    orig_out_dtype: Optional[str] = None
    orig_out_shape: Optional[str] = None
    error_message: Optional[str] = None


def create_pre_data_item(op, args, kwargs):
    return DataItem(args=args, kwargs=kwargs, origin_op=op)

def create_new_csv_row(data_item, op_name, disturb_factor, step, error_message=""):
    row = FreeBenchmarkCsvRow(
        step=step,
        api=op_name,
        disturb_factor=disturb_factor,
        error_message=error_message,
    )
    row.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if isinstance(data_item.origin_res, torch.Tensor):
        row.orig_out_dtype = str(data_item.origin_res.dtype)
        row.orig_out_shape = str(data_item.origin_res.shape)
    return row

def format_floats_in_dict(data, precision=7):
    if isinstance(data, dict):
        return {
            k: format_floats_in_dict(v, precision)
            for k, v in data.items()
        }
    elif isinstance(data, float):
        try:
            return round(data, precision)
        except:
            return data
    else:
        return data

def safe_args_copy(args, keep_in_autograd):
    if isinstance(args, dict):
        ret = {}
        for key, value in args.items():
            ret[key] = safe_args_copy(value, keep_in_autograd)
        return ret
    elif isinstance(args, (list, tuple)):
        return type(args)(safe_args_copy(arg, keep_in_autograd) for arg in args)
    elif isinstance(args, torch.Tensor):
        if keep_in_autograd:
            return args.clone()
        else:
            return copy.deepcopy(args.detach())
    else:
        try:
            if keep_in_autograd:
                return copy.copy(args)
            else:
                return copy.deepcopy(args)
        except:
            return args

def is_reduced_floating_point(tensor):
    # return True if tensor.dtype is FP8/FP16/BF16
    return torch.is_floating_point(tensor) and tensor.dtype not in [torch.float32, torch.float64]

def write_to_csv(row, header, save_path):
    if not row:
        return
    csv_exists = os.path.exists(save_path)
    mode = "a+" if csv_exists else "w+"
    with open(save_path, mode=mode) as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(header)
        writer.writerow(row)

