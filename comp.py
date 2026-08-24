import os
from collections import defaultdict, OrderedDict
from filecmp import dircmp
import glob
import pandas as pd
import numpy as np
import dill

import torch

from torchdump.advisor import Advisor
from torchdump.utils import read_from_custom, get_tensor_infos, TensorSummary, get_custom_op_map, \
    add_time_as_suffix, get_min_val, get_max_val, get_mean_val, get_norm_val, \
    add_custom_op_map, check_overflow_tensor, check_overflow_data, get_logger

from torchdump.evaluate import (
    calc_diff1,
    calc_diff2,
)

logger = get_logger()

# The accuracy standard refers to aw's.
# For LEVEL=1, failed when relative > 0.5
# For LEVEL=2, failed when cosine < 0.99 and max_abs > 0.001
# For LEVEL=2, failed when cosine < 0.9 or max_abs > 1
compare_thresholds = {
    "relative": 0.5,
    "cosine": 0.99,
    "max_abs": 0.001,
    "cosine_max": 0.9,
    "max_abs_max": 1
}

def set_compare_threshold(type, value):
    if type in compare_thresholds:
        compare_thresholds[type] = value
    else:
        logger.info(f"Threshold type '{type}' is not recognized. \
              Available types: {list(compare_thresholds.keys())}")

def get_relative_error(l_val, r_val, resolution):
    if l_val is None or r_val is None:
        return 0.0
    l_val = 0.0 if abs(l_val / resolution) < 10 else l_val
    r_val = 0.0 if abs(r_val / resolution) < 10 else r_val
    return abs(l_val - r_val) / (max(abs(l_val), abs(r_val)) + 1e-10)

def check_accuracy(tensor1, tensor2, overflow, compare_thresholds):
    def _reshape_value(tensor1, tensor2):
        if not tensor1.shape:
            if tensor1.dtype == bool:
                tensor1 = tensor1.float()
                tensor2 = tensor2.float()
            return tensor1, tensor2

        tensor1 = tensor1.reshape(-1).float()
        tensor2 = tensor2.reshape(-1).float()
        return tensor1, tensor2

    def _get_cosine_similarity(tensor1, tensor2):
        num = torch.dot(tensor1, tensor2).item()
        norm1 = torch.norm(tensor1).item()
        norm2 = torch.norm(tensor2).item()

        FLOAT_EPSILON = torch.finfo(torch.float).eps
        if norm1 <= FLOAT_EPSILON and norm2 <= FLOAT_EPSILON:
            return 1.0
        # Cannot compare by Cosine Similarity
        if norm1 <= FLOAT_EPSILON or norm2 <= FLOAT_EPSILON:
            return "NAN"

        cos = num / (norm1 * norm2)
        if np.isnan(cos):
            return "NAN"
        return 1.0 if float(cos) > 0.99999 else cos

    def _get_max_abs_error(tensor1, tensor2):
        abs_error = torch.abs(tensor1 - tensor2)
        return torch.max(abs_error).item()

    if overflow:
        return "NAN", "NAN", "NAN"
    if tensor1.dtype != tensor2.dtype:
        return "UNMATCHED", "NA", "NA"
    if not tensor1.shape or tensor1.numel() == 0:
        return "NONE", "NA", "NA"

    tensor1, tensor2 = _reshape_value(tensor1, tensor2)
    cosine = _get_cosine_similarity(tensor1, tensor2)
    max_abs = _get_max_abs_error(tensor1, tensor2)

    if cosine == "NAN":
        return "NAN", "NAN", "NAN"
    if cosine < compare_thresholds["cosine"] and max_abs > compare_thresholds["max_abs"]:
        return "FAILED", cosine, max_abs
    elif cosine < compare_thresholds["cosine_max"] or max_abs > compare_thresholds["max_abs_max"]:
        return "FAILED", cosine, max_abs

    return "PASSED", cosine, max_abs

# Some operator names do not need to match.
def escape4specials(item):
    return item.startswith('torch.Tensor.cpu.') or item.startswith('torch.Tensor.mlu.') \
        or item.startswith('torch.Tensor.cuda.')

def dir_check_helper(subdir_list, dcmp, dev0_custom_ops, dev1_custom_ops, suffix, leaf_dir=False):
    left_dir = dcmp.left
    right_dir = dcmp.right
    left_only = []
    for item in dcmp.left_only:
        flg = True
        for custom_op in dev0_custom_ops:
            if item.startswith(custom_op + '.') or escape4specials(item):
                flg = False
                break
        if flg:
            left_only.append(item)
            break    # only one is enough to report error
    right_only = []
    for item in dcmp.right_only:
        flg = True
        for custom_op in dev1_custom_ops:
            if item.startswith(custom_op + '.') or escape4specials(item):
                flg = False
                break
        if flg:
            right_only.append(item)
            break   # only one is enough to report error
    if left_only or right_only:
        if left_only:
            diff_file = left_only[0]
            diff_dir = left_dir
        else:
            diff_file = right_only[0]
            diff_dir = right_dir
        logger.warning(f"Diff file or directory '{diff_file}' found in '{diff_dir}'")
    msg = "please check or delete irrelevant files or directory."
    if dcmp.common_files and dcmp.common_dirs:
        raise RuntimeError("Both files '{}' and directory '{}' exist in {} and {}, {}".format( \
            dcmp.common_files[0], dcmp.common_dirs[0], left_dir, right_dir, msg))
    if dcmp.common_files:
        subdir_list.append(suffix)
        return
    for subdir in dcmp.common_dirs:
        if leaf_dir:
            raise RuntimeError(f"Invalid sub directory '{subdir}' found in '{left_dir}', {msg}")
        assert subdir.startswith('rank') or subdir.startswith('iter'), \
            f"The name of the subdirectory `{subdir}` of `{left_dir}` and `{right_dir}` " \
            "must starts with `rank` or `iter`."
        subsuffix = suffix + (f"/{subdir}" if suffix else str(subdir))
        dir_check_helper(subdir_list, dcmp.subdirs[subdir], dev0_custom_ops, dev1_custom_ops,
            subsuffix, True if subdir.startswith('rank') else False)

def check_dir(dcmp):
    dev0_custom_ops, dev1_custom_ops = read_from_custom()
    add_custom_op_map(["torch.ops.torch_mlu.amp_unscale", \
        "torch._amp_foreach_non_finite_check_and_unscale_"])
    add_custom_op_map(["torch._amp_foreach_non_finite_check_and_unscale_", \
        "torch.ops.torch_mlu.amp_unscale"])
    subdir_list = []
    dir_check_helper(subdir_list, dcmp, dev0_custom_ops, dev1_custom_ops, '')
    return subdir_list

custom_op_map = get_custom_op_map()
def check_name(lf_name, rf_name, index_strict=False):
    if lf_name == rf_name:
        return True
    lsplit = lf_name.split('.')
    rsplit = rf_name.split('.')
    if index_strict == True and lsplit[-4] != rsplit[-4]:
        return False
    if '.'.join(lsplit[-3:]) != '.'.join(rsplit[-3:]):
        return False
    lop_api = '.'.join(lsplit[:-4])
    rop_api = '.'.join(rsplit[:-4])
    if not ((lop_api == rop_api) \
         or (lop_api in custom_op_map and rop_api == custom_op_map[lop_api]) \
         or (escape4specials(lf_name) and escape4specials(rf_name))):
        return False
    return True

def check_tensor(dev0_dict, dev1_dict):
    dev0_name = dev0_dict["file_name"]
    dev1_name = dev1_dict["file_name"]
    if not check_name(dev0_name, dev1_name):
        return False
    if dev0_dict["t_num"] != dev1_dict["t_num"]:
        return False
    # If one device's api is torch.ops.torch_mlu.amp_unscale and the other device's api name is
    # torch._amp_foreach_non_finite_check_and_unscale_, loose the match condition, just compare tensors' num
    lsplit = dev0_name.split('.')
    rsplit = dev1_name.split('.')
    lop_api = '.'.join(lsplit[:-4])
    rop_api = '.'.join(rsplit[:-4])
    if (lop_api == "torch.ops.torch_mlu.amp_unscale" and \
        rop_api == "torch._amp_foreach_non_finite_check_and_unscale_") or \
       (lop_api == "torch._amp_foreach_non_finite_check_and_unscale_" and \
        rop_api == "torch.ops.torch_mlu.amp_unscale"):
        return True
    # else, compare the tensors' sizes
    dev0_t_shapes = dev0_dict["t_shapes"]
    dev1_t_shapes = dev1_dict["t_shapes"]
    for i in range(len(dev0_t_shapes)):
        if dev0_t_shapes[i] != dev1_t_shapes[i]:
            return False
    return True

def match_tensor(dev0_queue, dev1_queue):
    if len(dev0_queue) == 0 or len(dev1_queue) == 0:
        return -1, -1
    for dev1_index, dev1_tensor in enumerate(dev1_queue[0: -1]):
        if check_tensor(dev0_queue[-1], dev1_tensor):
            return len(dev0_queue) - 1, dev1_index
    if check_tensor(dev0_queue[-1], dev1_queue[-1]):
        return len(dev0_queue) - 1, len(dev1_queue) - 1
    for dev0_index, dev0_tensor in enumerate(dev0_queue[0: -1]):
        if check_tensor(dev0_tensor, dev1_queue[-1]):
            return dev0_index, len(dev1_queue) - 1
    return -1, -1

def get_unmatch_accuracy(result, dev_op_tensor, dev_dir, which_dev, show_mode):
    dev_name = dev_op_tensor.get("file_name", "NA")
    dev_tname = dev_op_tensor.get("thread_name", "NA")
    dev_index = dev_op_tensor.get("orig_idx", "NA")
    dev_file_path = os.path.join(dev_dir, dev_name)
    dev_all_info = torch.load(dev_file_path, map_location="cpu", pickle_module=dill)
    dev_tensors = get_tensor_infos()
    if len(dev_tensors) > 0:
        is_summary = isinstance(dev_tensors[0], TensorSummary)
    for j, dev_tensor in enumerate(dev_tensors):
        if dev_tensor.is_meta:
            continue
        res_item = []
        if which_dev == 1:
            res_item.append("NA")
        res_item.append(dev_index)
        res_item.append("NA")
        res_item.append(dev_name[:-2] + str(j))
        res_item.append("NA")
        res_item.append(dev_tname)
        res_item.append("NA")
        res_item.append(str(dev_tensor.dtype))
        res_item.append("NA")
        res_item.append(str(tuple(dev_tensor.size())))
        res_item.append("NA")
        if is_summary:
            max_val = dev_tensor.max_val
            min_val = dev_tensor.min_val
            mean_val = dev_tensor.mean_val
            norm_val = dev_tensor.norm_val
            overflow = check_overflow_data(max_val) or check_overflow_data(min_val)
            if show_mode == "overflow_only" and (not overflow):
                continue
        else:
            overflow = check_overflow_tensor(dev_tensor)
            if show_mode == "overflow_only" and (not overflow):
                continue
            max_val = get_max_val(dev_tensor)
            min_val = get_min_val(dev_tensor)
            mean_val = get_mean_val(dev_tensor)
            norm_val = get_norm_val(dev_tensor)
        res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)), [max_val]))
        res_item.append("NA")
        res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)), [min_val]))
        res_item.append("NA")
        res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)), [mean_val]))
        res_item.append("NA")
        res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)), [norm_val]))
        res_item.append("NA")
        res_item.extend(map(lambda v: str("No" if v is False else "Yes"), [overflow]))
        res_item.append("NA")
        res_item.append("NA")
        res_item.append("NA")
        res_item.append("NA") # max_abs
        res_item.append("NA") # cosine
        res_item.append("UNMATCHED") # result
        res_item.append(dev_all_info['stack'] if isinstance(dev_all_info, dict) and 'stack' in dev_all_info.keys() \
                else "NA")
        if which_dev == 0:
            res_item.append("NA")
        result.append(res_item)

def get_accuracy(
    result, dev0_name, dev1_name, left_dir, right_dir, show_mode,
    dev0_tname="NA", dev1_tname="NA", dev0_index="NA", dev1_index="NA",
):
    dev0_file_path = os.path.join(left_dir, dev0_name)
    dev1_file_path = os.path.join(right_dir, dev1_name)
    dev0_all_info = torch.load(dev0_file_path, map_location="cpu", pickle_module=dill)
    dev0_tensors = get_tensor_infos()
    dev1_all_info = torch.load(dev1_file_path, map_location="cpu", pickle_module=dill)
    dev1_tensors = get_tensor_infos()
    if len(dev0_tensors) != len(dev1_tensors):
        raise RuntimeError("The numbers of tensors found in '{}' and '{}' are not equal.". \
            format(dev0_file_path, dev1_file_path))
    if len(dev0_tensors) > 0:
        is_summary = isinstance(dev0_tensors[0], TensorSummary)
    # TODO(zhanchendi): may be possible to accelerate by multiple processes.
    for j, (dev0_tensor, dev1_tensor) in enumerate(zip(dev0_tensors, dev1_tensors)):
        if dev0_tensor.is_meta or dev1_tensor.is_meta:
            continue
        res_item = []
        res_item.append(dev0_index)
        res_item.append(dev1_index)
        res_item.append(dev0_name[:-2] + str(j))
        res_item.append(dev1_name[:-2] + str(j))
        res_item.append(dev0_tname)
        res_item.append(dev1_tname)
        res_item.append(str(dev0_tensor.dtype))
        res_item.append(str(dev1_tensor.dtype))
        res_item.append(str(tuple(dev0_tensor.size())))
        res_item.append(str(tuple(dev1_tensor.size())))
        if is_summary:
            l_max_val = dev0_tensor.max_val
            r_max_val = dev1_tensor.max_val
            l_min_val = dev0_tensor.min_val
            r_min_val = dev1_tensor.min_val
            l_mean_val = dev0_tensor.mean_val
            r_mean_val = dev1_tensor.mean_val
            l_norm_val = dev0_tensor.norm_val
            r_norm_val = dev1_tensor.norm_val
            diff1 = diff2 = cosine = "NA"
            max_abs = "NA"
            l_overflow = check_overflow_data(l_max_val) or check_overflow_data(l_min_val)
            r_overflow = check_overflow_data(r_max_val) or check_overflow_data(r_min_val)
            if show_mode == "overflow_only" and (not(l_overflow or r_overflow)):
                continue
            compare_result = "PASSED"
            if (
                dev0_tensor.size() != dev1_tensor.size()
                or dev0_tensor.dtype != dev1_tensor.dtype
            ):
                compare_result = "UNMATCHED"
            elif l_overflow or r_overflow:
                compare_result = "NAN"
            else:
                resolution = torch.finfo(dev0_tensor.dtype).resolution \
                    if dev0_tensor.dtype.is_floating_point else 1
                for l_val, r_val in zip([l_max_val, l_min_val, l_mean_val, l_norm_val],
                                        [r_max_val, r_min_val, r_mean_val, r_norm_val]):
                    if get_relative_error(l_val, r_val, resolution) > compare_thresholds["relative"]:
                        compare_result = "WARNING"
                        break
        else:
            l_overflow = check_overflow_tensor(dev0_tensor)
            r_overflow = check_overflow_tensor(dev1_tensor)
            if show_mode == "overflow_only" and (not(l_overflow or r_overflow)):
                continue
            l_max_val = get_max_val(dev0_tensor)
            r_max_val = get_max_val(dev1_tensor)
            l_min_val = get_min_val(dev0_tensor)
            r_min_val = get_min_val(dev1_tensor)
            l_mean_val = get_mean_val(dev0_tensor)
            r_mean_val = get_mean_val(dev1_tensor)
            l_norm_val = get_norm_val(dev0_tensor)
            r_norm_val = get_norm_val(dev1_tensor)
            try:
                diff1 = calc_diff1(
                    dev0_tensor.cpu().detach().to(torch.float64).numpy(), dev1_tensor.cpu().detach().to(torch.float64).numpy()
                )
                diff2 = calc_diff2(
                    dev0_tensor.cpu().detach().to(torch.float64).numpy(), dev1_tensor.cpu().detach().to(torch.float64).numpy()
                )
                compare_result, cosine, max_abs = check_accuracy(
                    dev0_tensor, dev1_tensor, l_overflow or r_overflow, compare_thresholds)
            except AssertionError:
                msg = "The sizes of compared tensors {} and {} are not equal.".format( \
                    dev0_tensor.size(), dev1_tensor.size())
                logger.warning("Calc the diff of the values of '{}' and '{}' " \
                    "in '{}' and '{}' failed! {}".format(res_item[0], res_item[1], dev0_file_path, dev1_file_path, msg))
                diff1 = diff2 = cosine = max_abs = "NA"
                compare_result = "UNMATCHED"
        res_item.extend(map(lambda v: str("NA" if v is None else round(v, 4)),
            [l_max_val, r_max_val, l_min_val, r_min_val, l_mean_val, r_mean_val, l_norm_val, r_norm_val]))
        res_item.extend(map(lambda v: str("No" if v is False else "Yes"), [l_overflow, r_overflow]))
        res_item.append(diff2 if isinstance(diff2, str) else str(round(diff2, 4)))
        res_item.append(diff1 if isinstance(diff1, str) else str(round(diff1, 4)))
        res_item.append(max_abs if isinstance(max_abs, str) else str(round(max_abs, 4)))
        res_item.append(cosine if isinstance(cosine, str) else str(round(cosine, 4)))
        res_item.append(compare_result)
        res_item.append(dev0_all_info['stack'] if isinstance(dev0_all_info, dict) and 'stack' in dev0_all_info.keys() \
            else "NA")
        res_item.append(dev1_all_info['stack'] if isinstance(dev1_all_info, dict) and 'stack' in dev1_all_info.keys() \
            else "NA")
        result.append(res_item)

columns = [
    "Dev0 Index", "Dev1 Index", "Dev0 Name", "Dev1 Name",
    "Dev0 Thread Name", "Dev1 Thread Name",
    "Dev0 Tensor Dtype", "Dev1 Tensor Dtype", "Dev0 Tensor Size", "Dev1 Tensor Size",
    "Dev0 Tensor Max", "Dev1 Tensor Max", "Dev0 Tensor Min", "Dev1 Tensor Min",
    "Dev0 Tensor Mean", "Dev1 Tensor Mean", "Dev0 Tensor Norm", "Dev1 Tensor Norm",
    "Dev0 Overflow Status", "Dev1 Overflow Status", "Diff2 Error", "Diff1 Error",
    "MaxAbs Error", "Cosine", "Result", "Dev0 Stack", "Dev1 Stack"
]
def run_single_match(
    cmp_res, lfs_list, rfs_list, left_dir, right_dir, has_same_tnames, show_mode="all"
):
    dev0_tensors_queue, dev1_tensors_queue = [], []
    while True:
        dev0_read_flag = dev1_read_flag = False
        if lfs_list:
            dev0_tensors_queue.append(lfs_list[0]); del lfs_list[0]; dev0_read_flag = True
        if rfs_list:
            dev1_tensors_queue.append(rfs_list[0]); del rfs_list[0]; dev1_read_flag = True
        if (not dev0_read_flag and not dev1_read_flag):
            break
        dev0_match_point, dev1_match_point = match_tensor(dev0_tensors_queue, dev1_tensors_queue)
        if dev0_match_point == -1 and dev1_match_point == -1:
            continue
        dev0_match_tensor = dev0_tensors_queue[dev0_match_point]
        dev1_match_tensor = dev1_tensors_queue[dev1_match_point]

        if has_same_tnames:
            dev0_unmatch_tensors = dev0_tensors_queue[0: dev0_match_point]
            dev1_unmatch_tensors = dev1_tensors_queue[0: dev1_match_point]

            for dev0_tensor in dev0_unmatch_tensors:
                get_unmatch_accuracy(cmp_res, dev0_tensor, left_dir, 0, show_mode)
            for dev1_tensor in dev1_unmatch_tensors:
                get_unmatch_accuracy(cmp_res, dev1_tensor, right_dir, 1, show_mode)

        try:
            get_accuracy(
                cmp_res, dev0_match_tensor["file_name"], dev1_match_tensor["file_name"],
                left_dir, right_dir, show_mode,
                dev0_match_tensor["thread_name"], dev1_match_tensor["thread_name"],
                dev0_match_tensor["orig_idx"], dev1_match_tensor["orig_idx"],
            )
        except RuntimeError:
            get_unmatch_accuracy(cmp_res, dev0_match_tensor, left_dir, 0, show_mode)
            get_unmatch_accuracy(cmp_res, dev1_match_tensor, right_dir, 1, show_mode)

        if has_same_tnames:
            del dev0_tensors_queue[0: dev0_match_point + 1]
            del dev1_tensors_queue[0: dev1_match_point + 1]
        else:
            del dev0_tensors_queue[dev0_match_point]
            del dev1_tensors_queue[dev1_match_point]

    if dev0_tensors_queue:
        for dev0_tensor in dev0_tensors_queue:
            get_unmatch_accuracy(cmp_res, dev0_tensor, left_dir, 0, show_mode)
    if dev1_tensors_queue:
        for dev1_tensor in dev1_tensors_queue:
            get_unmatch_accuracy(cmp_res, dev1_tensor, right_dir, 1, show_mode)

def reorder_op_list_strict(op_list):
    """
    Reorder op_list by grouping records belonging to the same operator instance.

    Each operator instance is identified by:
        (op_name, op_id, direction)

    Reordering rules:
    - All records of the same operator instance are kept together.
    - Within each operator instance: input record is placed before output record.
    - Different operator instances are ordered by their first appearance in the original op_list.
    - Thread ordering and interleaving are ignored.
    """
    def parse_file_name(file_name: str):
        """
        Parse file_name to extract operator metadata.

        Example file_name:
            torch.Tensor.item.920.fwd.input.pt

        Parsed fields:
            op_name   -> torch.Tensor.item
            op_id     -> 920
            direction -> fwd
            io        -> input / output
        """
        parts = file_name.split(".")
        return {
            "op_name": ".".join(parts[:-4]), "op_id": int(parts[-4]),
            "direction": parts[-3], "io": parts[-2],
        }

    op_groups = OrderedDict()
    for item in op_list:
        meta = parse_file_name(item["file_name"])
        # Example: ("torch.Tensor.item", 920, "fwd")
        key = (meta["op_name"], meta["op_id"], meta["direction"])

        if key not in op_groups:
            op_groups[key] = {"input": None, "output": None}

        if meta["io"] == "input":
            op_groups[key]["input"] = item
        elif meta["io"] == "output":
            op_groups[key]["output"] = item

    reordered = []
    for key, io_pair in op_groups.items():
        if io_pair["input"] is not None:
            reordered.append(io_pair["input"])
        if io_pair["output"] is not None:
            reordered.append(io_pair["output"])

    bwd_tnames = set()
    all_tnames = set()
    for idx, item in enumerate(reordered):
        item["orig_idx"] = idx
        meta = parse_file_name(item["file_name"])
        meta["direction"] == "bwd" and bwd_tnames.add(item.get("thread_name", "NA"))
        all_tnames.add(item.get("thread_name", "NA"))

    return reordered, bwd_tnames, all_tnames


def comp_dir_fuzzy_match(dir0, dir1, output_dir, subdir, show_mode = "all"):
    res_file = "compare_result_" + subdir.replace('/', '_')
    left_dir = os.path.join(dir0, subdir)
    right_dir = os.path.join(dir1, subdir)
    lfs_list = torch.load(os.path.join(left_dir, "dump_seqs.pt"), weights_only=False)[1]
    lfs_list, lfs_bwd_tnames, lfs_tnames = reorder_op_list_strict(lfs_list)
    rfs_list = torch.load(os.path.join(right_dir, "dump_seqs.pt"), weights_only=False)[1]
    rfs_list, rfs_bwd_tnames, rfs_tnames = reorder_op_list_strict(rfs_list)
    has_same_tnames = {
        t for t in (lfs_tnames & rfs_tnames) - lfs_bwd_tnames - rfs_bwd_tnames
        if not t.startswith("Dummy")
    }
    cmp_res = []

    if has_same_tnames:
        def bucket_op_list(op_list, bwd_tnames, shared_tnames):
            bucket = defaultdict(list)
            list_bwd, list_dummy, list_unmatched = [], [], []
            for item in op_list:
                tname = item.get("thread_name", "NA")
                if tname in bwd_tnames:
                    list_bwd.append(item)
                elif tname.startswith("Dummy"):
                    list_dummy.append(item)
                elif tname in shared_tnames:
                    bucket[tname].append(item)
                else:
                    list_unmatched.append(item)
            return bucket, list_bwd, list_dummy, list_unmatched

        lfs_bucket, lfs_list_bwd, lfs_list_dummy, lfs_list_unmatched = \
            bucket_op_list(lfs_list, lfs_bwd_tnames, has_same_tnames)
        rfs_bucket, rfs_list_bwd, rfs_list_dummy, rfs_list_unmatched = \
            bucket_op_list(rfs_list, rfs_bwd_tnames, has_same_tnames)

        # 1) Mixed matching for backward-thread records
        if lfs_list_bwd or rfs_list_bwd:
            run_single_match(
                cmp_res, lfs_list_bwd, rfs_list_bwd, left_dir, right_dir, False, show_mode
            )
        # 2) Mixed matching for remaining Dummy-prefixed threads (excluding backward-thread records)
        if lfs_list_dummy or rfs_list_dummy:
            run_single_match(
                cmp_res, lfs_list_dummy, rfs_list_dummy, left_dir, right_dir, False, show_mode
            )
        # 3) Bucketed matching for records whose thread names are shared between dev0 and dev1
        for tname in has_same_tnames:
            lfs_sub_list = lfs_bucket.get(tname, [])
            rfs_sub_list = rfs_bucket.get(tname, [])
            run_single_match(
                cmp_res, lfs_sub_list, rfs_sub_list, left_dir, right_dir, True, show_mode
            )
        # 4) Final mixed matching for remaining non-Dummy records with unmatched thread names
        if lfs_list_unmatched or rfs_list_unmatched:
            run_single_match(
                cmp_res, lfs_list_unmatched, rfs_list_unmatched, left_dir, right_dir, False, show_mode
            )

        res_file = add_time_as_suffix(res_file)
        res_df = pd.DataFrame(cmp_res, columns=columns)

        res_df["Dev0 Index"] = pd.to_numeric(res_df["Dev0 Index"], errors="coerce")
        res_df = res_df.sort_values("Dev0 Index", na_position="last", kind="mergesort")

        res_df.to_csv(os.path.join(output_dir, res_file), index=False, columns=res_df.columns[2:])

        return res_df

    run_single_match(
        cmp_res, lfs_list, rfs_list, left_dir, right_dir, has_same_tnames, show_mode
    )

    res_file = add_time_as_suffix(res_file)
    res_df = pd.DataFrame(cmp_res, columns=columns)
    res_df.to_csv(os.path.join(output_dir, res_file), index=False, columns=res_df.columns[2:])

    return res_df

def comp_dir(dir0, dir1, output_dir, subdir, show_mode = "all"):
    res_file = "compare_result_" + subdir.replace('/', '_')
    res_file = add_time_as_suffix(res_file)
    left_dir = os.path.join(dir0, subdir)
    right_dir = os.path.join(dir1, subdir)
    lfs_list = torch.load(os.path.join(left_dir, "dump_seqs.pt"), weights_only=False)[0]
    rfs_list = torch.load(os.path.join(right_dir, "dump_seqs.pt"), weights_only=False)[0]
    cmp_res = []
    for lf_name, rf_name in zip(lfs_list, rfs_list):
        lp = os.path.join(left_dir, lf_name)
        rp = os.path.join(right_dir, rf_name)
        if lf_name != rf_name:
            lsplit = lf_name.split('.')
            rsplit = rf_name.split('.')
            lop_api = '.'.join(lsplit[:-4])
            rop_api = '.'.join(rsplit[:-4])
            if not (((lop_api in custom_op_map and rop_api == custom_op_map[lop_api]) \
                or (escape4specials(lf_name) and escape4specials(rf_name))) \
                and '.'.join(lsplit[-4:]) == '.'.join(rsplit[-4:])):
                raise RuntimeError("Mismatched pt files '{}' and '{}' are found, "
                    "cannot compare two different operator sequences.".format(lp, rp))
            # the output of the amp unscale op between mlu and gpu is different.
            elif ((lop_api == "torch.ops.torch_mlu.amp_unscale" and \
                rop_api == "torch._amp_foreach_non_finite_check_and_unscale_") \
                or (lop_api == "torch._amp_foreach_non_finite_check_and_unscale_" and \
                rop_api == "torch.ops.torch_mlu.amp_unscale")) and lsplit[-2] == "output":
                continue
        get_accuracy(cmp_res, lf_name, rf_name, left_dir, right_dir, show_mode)

    res_df = pd.DataFrame(cmp_res, columns=columns)
    res_df.to_csv(os.path.join(output_dir, res_file), index=False)

    return res_df

def compare_dump(dir0, dir1, output_dir="./", fuzzy_match=False, show_mode="all", auto_analyze=True):
    assert isinstance(dir0, str) and isinstance(dir1, str), \
        f"The types of dir0:{type(dir0)} and dir1 must be str."
    assert os.path.exists(dir0), f"The directory of {dir0} is not exist."
    assert os.path.exists(dir1), f"The directory of {dir1} is not exist."
    assert isinstance(fuzzy_match, bool), f"The type of fuzzy_match:{type(fuzzy_match)} must be bool."
    assert isinstance(show_mode, str), f"The type of show_mode:{type(show_mode)} must be string."
    assert show_mode in ["all", "overflow_only"], "show_mode must be \"all\" or \"overflow_only\"."
    if fuzzy_match:
        logger.warning("This task uses fuzzy matching, which may affect the accuracy of the comparison.")

    dcmp = dircmp(dir0, dir1)
    subdir_list = check_dir(dcmp)
    if subdir_list and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in subdir_list:
        if fuzzy_match:
            res_df = comp_dir_fuzzy_match(dir0, dir1, output_dir, subdir, show_mode)
        else:
            res_df = comp_dir(dir0, dir1, output_dir, subdir, show_mode)

        if auto_analyze:
            advisor = Advisor(res_df, output_dir, subdir)
            advisor.analysis()
