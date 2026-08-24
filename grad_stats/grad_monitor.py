import torch
import os
import torch.distributed as dist
from collections import defaultdict
from torch.optim.optimizer import register_optimizer_step_pre_hook

from torchdump.utils import create_dirs, write_csv, remove_path, dist_operate, get_logger

from .grad_stats_utils import(
    level_interp,
    GradStatCsv,
    check_numeral_increase_list,
    save_npy,
    print_rank0_message,
    is_data_in_list
)

logger = get_logger()

class GradientMonitor:

    def __init__(self, config_json: dict,
                 ctx = None):
        self.task = config_json["task"]
        if self.task != "grad_stats":
            raise Exception(f"please, make sure to set task as grad_stats.task is {self.task}")
        self.level = config_json[self.task].get("grad_level", "L1")
        if self.level not in level_interp:
            raise Exception(f"please, make sure to set grad_level in {level_interp.keys()}")
        self._level_interp = level_interp[self.level]
        self._param_list = config_json[self.task].get("param_list", [])
        self._target_ranks = ctx.ranks
        logger.info(f"ranks are {self._target_ranks}")
        self._target_iters = ctx.iters
        logger.info(f"iters is {self._target_iters}")
        self._bounds = config_json[self.task].get("bounds", [-1, 0, 1])
        check_numeral_increase_list(self._bounds)
        self._output_path = ctx.output_dir
        # Some ranks may not dump, but also need paticipate in creating dump directory
        dist_operate(ctx, create_dirs, ctx.output_dir, ctx.ranks)
        self.current_rank = ctx.cur_rank
        self._step = -1
        self._param2name = defaultdict(str)
        self._ctx = ctx

    @property
    def output_path(self):
        return self._output_path

    @staticmethod
    def save_grad_direction(param_name, grad, save_path):
        if not os.path.exists(save_path):
            create_dirs(save_path)
        param_grad = grad.clone().detach()
        is_positive = param_grad > 0
        save_filepath = os.path.join(save_path, f"{param_name}.npy")
        save_npy(is_positive.cpu().numpy(), save_filepath)

    def monitor(self, model):
        print_rank0_message("parameter names:", self._ctx.cur_rank)
        for name, param in model.named_parameters():
            self._param2name[param] = name
            print_rank0_message(f"\t{name}", self._ctx.cur_rank)
        self._rank = self.current_rank if self.current_rank else 0
        self._hook_optimizer()

    def _hook_optimizer(self):
        def optimizer_pre_step_hook(optimizer, args, kargs):
            self._step += 1
            logger.info(f"grad_stats: optimizer step {self._step}")
            if not is_data_in_list(self._step, self._target_iters):
                return
            output_lines = []
            for param, param_name in self._param2name.items():
                if not is_data_in_list(param_name, self._param_list):
                    continue
                # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer.py
                # custom optimizer params have main_grad attribution.
                grad = param.main_grad if hasattr(param, "main_grad") else param.grad
                if grad is None:
                    logger.info(f"{param_name} hasn't grad function.")
                    continue
                if grad.dtype.is_complex:
                    logger.warning(f"dtype of {param_name} is {grad.dtype}, grad of {param_name} would be skipped.")
                    continue
                grad_info = GradStatCsv.generate_csv_line(param_name, self._level_interp, grad, self._bounds)
                output_lines.append(grad_info)
                if self._level_interp["have_grad_direction"]:
                    GradientMonitor.save_grad_direction(param_name, grad,
                                                        f'{self._output_path}/rank{self._rank}/step{self._step}')
            output_dirpath = os.path.join(self._output_path, f"rank{getattr(self, '_rank')}")
            if not os.path.isdir(output_dirpath):
                create_dirs(output_dirpath)
            output_path = os.path.join(output_dirpath, f"grad_summary_{self._step}.csv")
            if os.path.exists(output_path):
                logger.warning(f"{output_path} will be recoverd")
                remove_path(output_path)
            header_result = GradStatCsv.generate_csv_header(self._level_interp, self._bounds)
            output_lines.insert(0, header_result)
            write_csv(output_lines, output_path)
            logger.info(f"write grad data to {output_path}")

        register_optimizer_step_pre_hook(optimizer_pre_step_hook)
