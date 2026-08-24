from torchdump.utils import get_logger
from torchdump.free_benchmark.disturb_factor.to_cpu import ToCpuFactor
from torchdump.free_benchmark.disturb_factor.type_promotion import TypePromotionFactor


logger = get_logger()


def create_disturb_factor(disturb_factor, op_name):
    if disturb_factor == "to_cpu":
        return ToCpuFactor(op_name=op_name)
    elif disturb_factor == "type_promotion":
        return TypePromotionFactor(op_name=op_name)
    else:
        logger.error(f"[Free Benchmark] unknown disturb_factor: {disturb_factor}")
        raise Exception(f"unknown disturb_factor: {disturb_factor}")
