import ipaddress
import warnings
from dataclasses import dataclass
from typing import Optional


def init_task_config(task, json_config):
    task_config = BaseTaskConfig(task=task)
    json_task = json_config.get(task, {})
    if task == "online":
        old_ip_and_port = ""
        old_tcp_send_queue_size = 10
        # TODO(): deprecated outside online args and move them into "online" dict
        if "ip_and_port" in json_config:
            warnings.warn("\"ip_and_port\" of \"dump\" is deprecated, please set it in \"online\" dict.", DeprecationWarning)
            old_ip_and_port = json_config["ip_and_port"]
        if "tcp_send_queue_size" in json_config:
            warnings.warn("\"tcp_send_queue_size\" of \"dump\" is deprecated, please set it in \"online\" dict.", DeprecationWarning)
            old_tcp_send_queue_size = json_config["tcp_send_queue_size"]
        ip_and_port = json_task.get("ip_and_port", old_ip_and_port)
        tcp_send_queue_size = json_task.get("tcp_send_queue_size", old_tcp_send_queue_size)
        task_config = OnlineTaskConfig(
                task=task,
                ip_and_port=ip_and_port,
                tcp_send_queue_size=tcp_send_queue_size
        )
    elif task == "overflow_check":
        overflow_nums = json_task.get("overflow_nums", 1)
        task_config = OverflowCheckTaskConfig(
                task=task,
                overflow_nums=overflow_nums
        )
    elif task == "free_benchmark":
        mode = json_task.get("mode", "check")
        stage = json_task.get("stage", "forward")
        disturb_factor = json_task.get("disturb_factor", "type_promotion")
        task_config = FreeBenchmarkTaskConfig(
                task=task,
                mode=mode,
                stage=stage,
                disturb_factor=disturb_factor
        )
    return task_config


@dataclass
class BaseTaskConfig:
    task: str = "default"


@dataclass
class OnlineTaskConfig(BaseTaskConfig):
    ip_and_port: Optional[str] = None
    tcp_send_queue_size: Optional[int] = None

    def __post_init__(self):
        assert isinstance(self.ip_and_port, str), \
            f"The types of ip_and_port:{self.ip_and_port} must be string."
        assert isinstance(self.tcp_send_queue_size, int), f"The type of tcp_send_queue_size:{self.tcp_send_queue_size} must be int."
        self.ip_addr = ""
        self.port = -1
        if self.ip_and_port:
            ip_port_lst = self.ip_and_port.rsplit(':', 1)
            assert len(ip_port_lst) == 2, "ip_and_port must follow the format: xx.xx.xx.xx:port for ipv4 or xx:xx:xx:xx:xx:xx:xx:xx:port for ipv6"
            if not self.is_valid_ipv4_addr(ip_port_lst[0]) and not self.is_valid_ipv6_addr(ip_port_lst[0]):
                raise Exception(f"The ip you passed is not a valid IPAddress: {ip_port_lst[0]}")
            if not (1024 < int(ip_port_lst[1]) <= 65535):
                raise Exception(f"The port num must be in range 1025-65535, but you passed is: {ip_port_lst[1]}")
            self.ip_addr, self.port = ip_port_lst[0], int(ip_port_lst[1])

    def is_valid_ipv4_addr(self, ip):
        try:
            ipaddress.IPv4Address(ip)
            return True
        except ipaddress.AddressValueError:
            return False

    def is_valid_ipv6_addr(self, ip):
        try:
            ipaddress.IPv6Address(ip)
            return True
        except ipaddress.AddressValueError:
            return False


@dataclass
class OverflowCheckTaskConfig(BaseTaskConfig):
    overflow_nums: Optional[int] = None


@dataclass
class FreeBenchmarkTaskConfig(BaseTaskConfig):
    mode: Optional[str] = None
    stage: Optional[str] = None
    disturb_factor: Optional[str] = None

    def __post_init__(self):
        assert self.mode is not None and self.mode in ["check", "verify"], \
                f"\"mode\" must be one of from [\"check\", \"verify\"], but current value is {self.mode}."
        assert self.stage is not None and self.stage in ["forward", "backward", "all"], \
                f"\"stage\" must be one of from [\"forward\", \"backward\", \"all\"], but current value is {self.stage}."
        assert self.disturb_factor is not None and self.disturb_factor in ["type_promotion", "to_cpu"], \
                f"\"disturb_factor\" must be one of from [\"type_promotion\", \"to_cpu\"], but current value is {self.disturb_factor}."
        assert not (self.mode == "verify" and self.stage == "backward"), \
                "\"mode\" and \"stage\" can not be \"verify\" and \"backward\" at the same time."

