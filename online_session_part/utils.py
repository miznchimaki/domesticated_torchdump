import os
import torch
import dill
import pickle

from collections import namedtuple
from filelock import FileLock

from torchdump.utils import (
        stat_tensors_info,
        get_logger,
)

logger = get_logger()

# data format for online communication
ApiInfo = namedtuple("ApiInfo",
                     ["name", "args", "kwargs", "out", "autocast_config", "grad_enable", "allow_tf32", "rank"],
                     defaults=["unknown", None, None, None, None, False, None, 0])

# the delimiter to identify ApiInfo boundaries in bytes
DELIMITER = b"---APIBOUND---"


def online_dump_save(dump_cont, op_name, is_input, result_dir, rank):
    '''
    Used for dumping data (following the offline format) when online comparison is not pass.

    return False if dump failed.
    '''
    output_dir = os.path.join(result_dir, "error_case_dump_dir", f"rank{rank}")
    output_dir = os.path.realpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_suffix = ".fwd.input.pt" if is_input else ".fwd.output.pt"
    dump_path = os.path.join(output_dir, op_name + file_suffix)
    try:
        torch.save(dump_cont, dump_path, pickle_module=dill,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Error occurs when dumping error case for {op_name+file_suffix}, skip it: {e}")
        if os.path.exists(dump_path):
            os.remove(dump_path)
        return False

    # record dump file
    dump_seqs_path = os.path.join(output_dir, "dump_seqs.pt")
    lock_path = dump_seqs_path + ".lock"
    with FileLock(lock_path):
        _dump_seqs = {}
        if not os.path.exists(dump_seqs_path):
            _dump_seqs[output_dir] = [[], []]
        else:
            _dump_seqs[output_dir] = torch.load(dump_seqs_path, weights_only=False)
        _dump_seqs[output_dir][0].append(op_name + file_suffix)
        tensors_info_dict = {}
        tensors_info_dict["file_name"] = op_name + file_suffix
        tensors_info_dict["t_shapes"] = []
        keys_to_check = ["args", "kwargs", "res"]
        for key in keys_to_check:
            if key in dump_cont.keys():
                stat_tensors_info(dump_cont[key], tensors_info_dict["t_shapes"])
        tensors_info_dict["t_num"] = len(tensors_info_dict["t_shapes"])
        _dump_seqs[output_dir][1].append(tensors_info_dict)
        torch.save(_dump_seqs[output_dir], dump_seqs_path, pickle_module=dill,
                pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return True

def check_in_whitelist(whitelist, api_fullname):
    if not whitelist:
        return True
    for api in whitelist:
        if api in api_fullname.lower():
            return True
    return False

def check_in_blacklist(blacklist, api_fullname):
    if not blacklist:
        return False
    for api in blacklist:
        if api in api_fullname.lower():
            return True
    return False
