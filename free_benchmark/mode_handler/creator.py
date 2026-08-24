from torchdump.utils import get_logger
from torchdump.free_benchmark.mode_handler.verify_mode import VerifyModeHandler
from torchdump.free_benchmark.mode_handler.check_mode import CheckModeHandler


logger = get_logger()


def create_mode_handler(mode, op_name, stage, output_dir):
    if mode == "check":
        return CheckModeHandler(op_name=op_name, stage=stage, output_dir=output_dir)
    elif mode == "verify":
        return VerifyModeHandler(op_name=op_name, stage=stage)
    else:
        logger.error(f"[Free Benchmark] unknown mode: {mode}")
        raise Exception(f"unknown disturb_factor: {mode}")
