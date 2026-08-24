import os
import csv
import warnings
import random
import torch
import torch.multiprocessing as mp
from queue import Empty
import pandas as pd
from filelock import FileLock

from torchdump.utils import (
    import_api_from_str,
    get_logger,
)
from torchdump.online_session_part.utils import online_dump_save

from .task import ReplayTask
from .utils import (
    set_current_device,
    clone_to_device,
    read_from_custom,
    get_custom_api_dict,
    replay_on_device_per_task,
    try_import_migration,
    rename_privateuse1_backend_for_mlu,
)

# ensure each subprocess in both offline and online replay to call gpu_migration
try_import_migration()

logger = get_logger()

class OnlineReplayTask(ReplayTask):
    def __init__(self, api_data, api_dict=None):
        self.api_data = api_data
        self.api_dict = api_dict
        self.rank = api_data.rank
        api_name_lst = api_data.name.rsplit('.', maxsplit=1)
        self.api, self.id = api_name_lst[0], int(api_name_lst[1])

        # online replay doesn't run backward
        self.bwd_access = False
        self.bwd_inputs = None
        self.bwd_outputs = None

    def get_op_and_inputs(self, map_location):
        self.api = self.api_dict.get(map_location, self.api) if self.api_dict is not None else self.api
        op = import_api_from_str(self.api, map_location)
        args = clone_to_device(self.api_data.args, map_location)
        kwargs = clone_to_device(self.api_data.kwargs, map_location)
        logger.debug("op: {}".format(self.api))
        logger.debug("args: {}".format(args))
        logger.debug("kwargs: {}".format(kwargs))
        logger.debug("autocast_config: {}".format(self.api_data.autocast_config))
        logger.debug("grad_enable: {}".format(self.api_data.grad_enable))
        logger.debug("allow_tf32: {}".format(self.api_data.allow_tf32))

        return op, args, kwargs, \
                self.api_data.autocast_config, self.api_data.grad_enable, \
                self.api_data.allow_tf32

    def get_outputs(self, map_location=None):
        out = self.api_data.out
        return clone_to_device(out, map_location)


def replay_on_device_online(
        device_id,
        data,
        custom_op_map,
        custom_ops_dev,
        cmd_args,
):
    set_current_device(cmd_args.device, device_id)

    # processing custom op
    api = data.name.rsplit('.', maxsplit=1)[0]
    api_dict = get_custom_api_dict(api, custom_op_map, custom_ops_dev)

    # create task node and run
    task = OnlineReplayTask(data, api_dict=api_dict)

    result, failed_case, need_save, need_dump = replay_on_device_per_task(
            task, cmd_args.device, cpu_baseline=cmd_args.cpu_baseline,
            overflow_check=cmd_args.overflow_check, configs=cmd_args.configs
            )

    # save failed case
    if failed_case is not None:
        logger.error(f"rank:{task.rank}, id:{task.id}, api:{task.api}, failure reason:{failed_case[-1]}")
        failed_case_dict = {
            "rank": task.rank,
            "id": task.id,
            "op": task.api,
            "failure reason": failed_case[-1],
        }
        df_fail = pd.DataFrame({key: [value] for key, value in failed_case_dict.items()})
        fail_path = os.path.join(cmd_args.result_dir, "fail_case.csv")
        lock_fail_path = fail_path + ".lock"
        with FileLock(lock_fail_path):
            if not os.path.exists(fail_path):
                df_fail.to_csv(fail_path, index=False)
            else:
                df_fail.to_csv(fail_path, mode='a', index=False, header=False)

    # dump cases that is not pass or is overflow, and support to replay offline
    if need_dump:
        dump_cont_input = {"args": data.args,
                     "kwargs": data.kwargs,
                     "autocast_config": data.autocast_config,
                     "grad_enable": data.grad_enable,
                     "allow_tf32": data.allow_tf32,
                     }
        if online_dump_save(dump_cont_input, data.name, True, cmd_args.result_dir, data.rank):
            dump_cont_output = {"res": data.out}
            online_dump_save(dump_cont_output, data.name, False, cmd_args.result_dir, data.rank)

    # save results that is not pass or choose overflow
    if need_save:
        for diff_name in ["diff1", "diff2", "diff3"]:
            result.pop(f"bwd {diff_name}", None)
        if cmd_args.print_result:
            print(f"mismatched op - rank:{task.rank}, id:{task.id}, api:{task.api}, fwd diff1:{result['fwd diff1']}, fwd diff2:{result['fwd diff2']}")

        df_result = pd.DataFrame({key: [value] for key, value in result.items()})
        result_path = os.path.join(cmd_args.result_dir, "online_compare_result.csv")
        lock_path = result_path + ".lock"
        with FileLock(lock_path):
            df_result.to_csv(result_path, mode='a', index=False, header=False)


def replay_on_device_online_loop(device_id, data_queue, cmd_args):
    # avoid queue.get() raising error
    rename_privateuse1_backend_for_mlu()

    # disable FutureWarning because too much information is printed
    warnings.filterwarnings("ignore", category=FutureWarning)

    # read custom ops yaml
    custom_op_map, custom_ops_dev = read_from_custom()
    while True:
        data = data_queue.get()
        if data == "REPLAY_END":
            break
        replay_on_device_online(device_id, data, custom_op_map, custom_ops_dev, cmd_args)


class OnlineReplayConsumer:
    def __init__(self, num_workers=1, queue_maxsize=16, cmd_args=None) -> None:
        # create empty result csv
        result_path = os.path.join(cmd_args.result_dir, "online_compare_result.csv")
        header = ["rank", "id", "op", "fwd diff1", "fwd diff2"]
        if cmd_args.configs.get("dynamic_accuracy_check", None):
            header.append("fwd diff3")
        if not os.path.exists(result_path):
            with open(result_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        # spawn multiple consumer processes
        self.processes = []
        ctx = mp.get_context("spawn")
        self.data_queues = [ctx.Queue(maxsize=queue_maxsize) 
                            for _ in range(num_workers)]
        for idx, queue in enumerate(self.data_queues):
            p = ctx.Process(target=replay_on_device_online_loop,
                           args=(idx, queue, cmd_args))
            p.start()
            self.processes.append(p)
        logger.info("start to run online replay...")

    def terminate(self):
        for queue in self.data_queues:
            queue.put("REPLAY_END")

        for p in self.processes:
            p.join()
        logger.info("online replay finished.")

    def cache_data(self, data):
        # use Least Loaded distribution fisrt
        min_queue = min(self.data_queues, key=lambda q: q.qsize())
        min_size = min_queue.qsize()
        min_idx_lst = [idx for idx,q in enumerate(self.data_queues) if q.qsize() == min_size]
        if min_idx_lst:
            min_idx = random.choice(min_idx_lst)
            min_queue = self.data_queues[min_idx]
        min_queue.put(data)
