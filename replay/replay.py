import argparse
import csv
import os
import time
import warnings
import pandas as pd
from collections import defaultdict
import shutil
import re
import json

import torch

from torchdump.utils import (
    load_json,
    get_logger,
)
from torchdump.online_session_part.utils import ApiInfo, check_in_whitelist, check_in_blacklist

from .task import ReplayTask
from .utils import (
    get_device_count,
    read_from_custom,
    get_custom_api_dict,
    set_current_device,
    replay_on_device_per_task,
    rename_privateuse1_backend_for_mlu,
)
from .online import OnlineReplayConsumer

logger = get_logger()

def clear_directory(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def readArgs():
    parser = argparse.ArgumentParser(description="replay operator for precision comparison")
    parser.add_argument(
        "--inputs",
        type=str,
        default="./dump_dir",
        help="input file for replay op",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mlu", "cuda"],
        required=True,
        help="Specify the device of the node replay",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="--tol will be deprecated. Please set tolerance in config.json",
    )
    parser.add_argument(
        "--config-path",
        dest="config_path",
        default="",
        type=str,
        help="The path of config.json")
    parser.add_argument(
        "--cpu-baseline",
        action="store_true",
        help="Use the CPU's running results as a baseline",
    )
    parser.add_argument(
        "--random-data",
        action="store_true",
        help="Use random data as input. Enforce cpu-baseline to True when enable random-data",
    )
    parser.add_argument(
        "--overflow-check",
        action="store_true",
        help="Check if the outputs overflows",
    )
    parser.add_argument(
        "--num-dev",
        type=int,
        default=1,
        help="Specify the num of device to replay",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="List of ids to be replayed, seperated by commas, default to replay all",
    )
    parser.add_argument(
        "--filter",
        action="append",
        help="filter op by regexp",
    )
    parser.add_argument(
        "--print-result",
        action="store_true",
        help="Print diff",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="./result_dir",
        help="Store info to file",
    )
    parser.add_argument(
        "--online-mode",
        type=str,
        choices=["no", "nfs", "tcp"],
        default="no",
        help="Specify the method of online replay mode. If choose \"tcp\", also need to specify \"--port\"",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=-1,
        help="Specify the port number of TCP server to listen on",
    )

    args, _ = parser.parse_known_args()
    if args.random_data:
        args.cpu_baseline = True
    args.ids = list(map(int, args.ids.split(","))) if args.ids is not None else []
    args.filter = "|".join(args.filter or [r"."])

    # read configs of replay
    args.configs = load_json(args.config_path)["replay"]

    if args.tol is not None:
        warnings.warn(
            "--tol will be deprecated. Please set tolerance in config.json", DeprecationWarning
        )
        args.configs["diff1_static_threshold"] = args.tol
        args.configs["diff2_static_threshold"] = args.tol

    if args.configs.get("dynamic_accuracy_check", None):
        assert args.device in ["cuda", "cpu"], f"for dynamic accuracy check, device must be cuda or cpu, but got {args.device}"
        if args.cpu_baseline:
            logger.warning("for dynamic accuracy check, cpu_baseline will not take effect")

    if args.online_mode == "tcp":
        assert args.port != -1, "Need to specify \"--port\" argument when choosing tcp mode."

    return args

def extract_nodes_info(inputs_dir, ids=[], filter="."):
    nodes_info_all_rank = defaultdict(lambda: defaultdict(list))
    dump_seqs_file = os.path.join(inputs_dir, "dump_seqs.pt")
    if os.path.exists(dump_seqs_file):
        # dump_dir/dump_seqs.pt
        leaf_dir = os.path.basename(os.path.realpath(inputs_dir))
        rank = leaf_dir if leaf_dir.startswith("rank") else "rank0"
        files_seqs = torch.load(dump_seqs_file, weights_only=False)[1]
        dirname = os.path.dirname(dump_seqs_file)
        nodes_info = defaultdict(list)
        for item in files_seqs:
            pt_file = item["file_name"]
            # TODO: [PYTORCH-14945] fix this when online mode support cpu scalar dump.
            is_run_on_cpu = item["is_run_on_cpu"] if "is_run_on_cpu" in item else True
            idx = int(pt_file.split(".")[-4])
            if len(ids) > 0 and idx not in ids:
                continue
            if not re.search(filter, pt_file, re.I):
                continue
            nodes_info[idx].append((os.path.join(dirname, pt_file), is_run_on_cpu))
        nodes_info_all_rank[rank] = nodes_info
    else:
        # dump_dir/[iter{0~9}*/][rank{0~65536}/]dump_seqs.pt
        subdirs = [f for f in os.listdir(inputs_dir)]
        subdirs.sort()
        for sd in subdirs:
            subdir_nodes_info = extract_nodes_info(os.path.join(inputs_dir, sd), ids, filter)
            for rank, nodes_info in subdir_nodes_info.items():
                for idx, ni in nodes_info.items():
                    nodes_info_all_rank[rank][idx].extend(ni)

    return nodes_info_all_rank

def preprocess_nodes(nodes_info_all_rank):
    custom_op_map, custom_ops_dev = read_from_custom()

    def create_nodes(rank, id, files):
        op_wise_files = defaultdict(list)
        for file, run_on_cpu in files:
            op_name = '.'.join(file.split(".")[0:-4])
            op_wise_files[op_name].append((file, run_on_cpu))

        ret_tasks = []
        for op_file in op_wise_files.values():
            fwd_input = None
            fwd_output = None
            bwd_input = None
            bwd_output = None
            op_name = None
            run_on_cpu_fwd = None
            run_on_cpu_bwd = None
            for file, run_on_cpu in op_file:
                if file.endswith("fwd.input.pt"):
                    fwd_input = file
                    run_on_cpu_fwd = run_on_cpu
                elif file.endswith("fwd.output.pt"):
                    fwd_output = file
                elif file.endswith("bwd.input.pt"):
                    bwd_input = file
                    run_on_cpu_bwd = run_on_cpu
                elif file.endswith("bwd.output.pt"):
                    bwd_output = file
                else:
                    logger.error("file {} is invalid".format(file))
                    exit(1)
            node_info = os.path.basename(fwd_input)
            api = ".".join(node_info.split(".")[0:-4])
            api_dict = get_custom_api_dict(api, custom_op_map, custom_ops_dev)
            # check whether api is distributed api
            check_name_space = '.'.join(node_info.split('.')[0:-5])
            if check_name_space == "torch.distributed" or check_name_space == "torch.distributed.distributed_c10d" \
                or check_name_space == "torch.ops.c10d":
                ret_tasks.append(None)
                continue

            ret_tasks.append(ReplayTask(rank,
                                        id,
                                        api,
                                        fwd_input,
                                        run_on_cpu_fwd,
                                        fwd_output,
                                        bwd_input,
                                        run_on_cpu_bwd,
                                        bwd_output,
                                        api_dict))
        return ret_tasks
    replay_tasks = defaultdict(list)
    for rank, nodes_info in nodes_info_all_rank.items():
        rank = int(rank.replace("rank", ""))
        for idx, pt_files in nodes_info.items():
            nodes = create_nodes(rank, idx, pt_files)
            for node in nodes:
                # if node is distributed api, skip replay
                if node is not None:
                    replay_tasks[rank].append(node)
    return replay_tasks

def map_to_device(replay_tasks, num_device):
    device_tasks = defaultdict(list)
    for rank, tasks in replay_tasks.items():
        for idx in range(len(tasks)):
            device_id = idx % num_device
            device_tasks[device_id].append(tasks[idx])
    return device_tasks


def replay_on_device(
    device_id,
    device_type,
    device_tasks,
    result_dir,
    random_data=False,
    cpu_baseline=False,
    overflow_check=False,
    configs={},
):
    # NOTE: when replay in cuda and input of op is torch.device("mlu"), will raise error:
    # "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: mlu"
    rename_privateuse1_backend_for_mlu()

    # disable FutureWarning because too much information is printed
    warnings.filterwarnings("ignore", category=FutureWarning)

    from . import config

    config.random_data = random_data
    config.device = device_type
    from .. import utils

    utils.disable_add_tensor_info = True

    tasks = device_tasks[device_id]
    set_current_device(device_type, device_id)
    local_result = {
        "rank": [],
        "id": [],
        "op": [],
        "fwd diff1": [],
        "fwd diff2": [],
        "fwd diff3": [],
        "bwd diff1": [],
        "bwd diff2": [],
        "bwd diff3": [],
    }
    if overflow_check:
        local_result["overflow"] = []
    failed_result = {
        "rank": [],
        "id": [],
        "op": [],
        "failure reason": [],
    }
    for task in tasks:
        result, failed_case, need_save, _ = replay_on_device_per_task(
                task, device_type, cpu_baseline=cpu_baseline,
                overflow_check=overflow_check, configs=configs
                )
        if failed_case is not None:
            failed_result["rank"].append(failed_case[0])
            failed_result["id"].append(failed_case[1])
            failed_result["op"].append(failed_case[2])
            failed_result["failure reason"].append(failed_case[3])
        elif need_save:
            local_result["rank"].append(result["rank"])
            local_result["id"].append(result["id"])
            local_result["op"].append(result["op"])
            for diff_name in ["diff1", "diff2", "diff3"]:
                local_result[f"fwd {diff_name}"].append(
                        result.get(f"fwd {diff_name}", None))
                local_result[f"bwd {diff_name}"].append(
                        result.get(f"bwd {diff_name}", None))
            if overflow_check:
                local_result["overflow"].append(
                        result.get("overflow", None))

    if configs.get("dynamic_accuracy_check", None):
        local_result.pop("overflow", None)
    else:
        local_result.pop("fwd diff3", None)
        local_result.pop("bwd diff3", None)

    df_failed = pd.DataFrame(failed_result)
    df_failed.to_csv(
        os.path.join(result_dir, device_type + str(device_id) + "_fail_case.csv"),
        index=False,
    )

    df = pd.DataFrame(local_result)
    df.to_csv(
        os.path.join(result_dir, device_type + str(device_id) + "_result.csv"),
        index=False,
    )
    return df, df_failed

def offline_replay(max_num_device, args):
    nodes_info = extract_nodes_info(args.inputs, args.ids)
    replay_tasks = preprocess_nodes(nodes_info)
    device_tasks = map_to_device(replay_tasks, max_num_device)

    result = None
    if max_num_device > 1:
        import torch.multiprocessing as mp

        mp.spawn(
            replay_on_device,
            args=(
                args.device,
                device_tasks,
                args.result_dir,
                args.random_data,
                args.cpu_baseline,
                args.overflow_check,
                args.configs,
            ),
            nprocs=max_num_device,
            join=True,
            daemon=True,
        )
    else:
        result, _ = replay_on_device(
            0,
            args.device,
            device_tasks,
            args.result_dir,
            args.random_data,
            args.cpu_baseline,
            args.overflow_check,
            args.configs,
        )

    logger.info("replay finished.")

    # load fail case of subprocess
    fail_case = []
    for csv_f in os.listdir(args.result_dir):
        if csv_f.endswith("_fail_case.csv"):
            csv_f = os.path.join(args.result_dir, csv_f)
        else:
            continue
        fail_result = pd.read_csv(csv_f)
        if not fail_result.empty:
            fail_case.append(fail_result)

    # load compare result of subprocess
    result = []
    for csv_f in os.listdir(args.result_dir):
        if csv_f.endswith("_result.csv"):
            csv_f = os.path.join(args.result_dir, csv_f)
        else:
            continue
        local_result = pd.read_csv(csv_f)
        if not local_result.empty:
            result.append(local_result)

    clear_directory(args.result_dir)

    if len(fail_case) > 0:
        fail_case = pd.concat(fail_case, ignore_index=True).sort_values(by=["rank", "id", "op"])
        fail_case.to_csv(os.path.join(args.result_dir, "fail_case.csv"), index=False)

    if len(result) > 0:
        result = pd.concat(result, ignore_index=True).sort_values(by=["rank", "id", "op"])
        result.to_csv(os.path.join(args.result_dir, "compare_result.csv"), index=False)
    else:
        # create empty result csv, keep consistent with online
        result_path = os.path.join(args.result_dir, "compare_result.csv")
        header = ["rank", "id", "op", 
                  "fwd diff1", "fwd diff2", "fwd diff3",
                  "bwd diff1", "bwd diff2", "bwd diff3",
                  ]
        if not args.configs.get("dynamic_accuracy_check", None):
            header.remove("fwd diff3")
            header.remove("bwd diff3")
        if not os.path.exists(result_path):
            with open(result_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    # print fail case
    if len(fail_case) > 0:
        logger.error(f"fail cases as follows:\n {fail_case.reset_index(drop=True)}")
        logger.error("please use \'--ids={}\' to replay only fail tasks".format(
          ",".join([str(id) for id in list(fail_case['id'])])))

    if len(result) > 0:
        if args.print_result:
            print("\nmismatched ops as follow:\n", result.reset_index(drop=True))
    else:
        print("\nno mismatch!")

def online_replay(max_num_device, args):
    # see NOTE: when replay in cuda and input of op is torch.device("mlu"), will raise error:
    rename_privateuse1_backend_for_mlu()

    consumer = OnlineReplayConsumer(num_workers=max_num_device, cmd_args=args)
    whitelist = [op.lower() for op in args.configs.get("online_whitelist", [])]
    blacklist = [op.lower() for op in args.configs.get("online_blacklist", [])]
    load_map_location = args.device if args.device in ["cuda", "mlu"] else "cpu"

    def check_in_filter(api_full_name):
        api_idx = int(api_full_name.rsplit('.', maxsplit=1)[1])
        if check_in_whitelist(whitelist, api_full_name) and \
                not check_in_blacklist(blacklist, api_full_name) and \
                (not args.ids or api_idx in args.ids):
            return True
        return False

    # nfs mode
    if args.online_mode == "nfs":
        from torchdump.online_session_part.nfs_agent import NFSAgent
        if not os.path.exists(args.inputs):
            os.makedirs(args.inputs)
        agent = NFSAgent(
            nfs_dir=args.inputs, load_map_location=load_map_location)
        while True:
            data = agent.recv()
            if data == "ONLINE_END":
                # agent.can_stop() returns True means server's connection_nums has been reduced to zero
                if agent.can_stop():
                    consumer.terminate()
                    break
                continue
            if not isinstance(data, ApiInfo):
                continue
            if check_in_filter(data.name):
                consumer.cache_data(data)
    # tcp mode
    elif args.online_mode == "tcp":
        from torchdump.online_session_part.tcp_agent import TCPAgent
        agent = TCPAgent(
            port=args.port, recv_map_location=load_map_location)
        remain_cnt = -1
        while True:
            if remain_cnt == 0:
                consumer.terminate()
                break
            data = agent.recv()
            if remain_cnt > 0:
                remain_cnt -= 1
            if data == "ONLINE_END":
                # wait for seconds to ensure the client has completed loseConnection() first.
                time.sleep(3)
                # agent.can_stop() returns True means server's connection_nums has been reduced to zero
                if agent.can_stop():
                    # need to recv all data before terminating
                    remain_cnt = agent.length_of_recv_queue()
                continue
            if not isinstance(data, ApiInfo):
                continue
            if check_in_filter(data.name):
                consumer.cache_data(data)

def main():
    start_time = time.time_ns()
    args = readArgs()

    device_count = get_device_count(args.device)
    max_num_device = min(device_count, args.num_dev)
    logger.info(f"device num used to replay : {str(max_num_device)}")

    clear_directory(args.result_dir)

    # offline replay
    if args.online_mode == "no":
        offline_replay(max_num_device, args)
    # online replay
    else:
        online_replay(max_num_device, args)
    logger.info("execution take time: {} s".format((time.time_ns() - start_time) / 1e9))
