import time
import torch
import io
import dill
import pickle
import queue

from torchdump.utils import get_logger

from .agent import OnlineAgent
from .utils import ApiInfo
from .tcp_server import TCPServer
from .tcp_client import TCPClient

logger = get_logger()

class TCPAgent(OnlineAgent):
    '''A method of Online Session Part.
    In LAN communication scenario, use socket over TCP/IP to transport PyTorch Tensors.
    '''
    def __init__(self, port, ip_addr=None, tcp_send_queue_size=10, recv_map_location="cpu") -> None:
        self.recv_map_location = recv_map_location
        if ip_addr is None:
            self.session = TCPServer(port)
            self.recv_queue = self.session.get_recv_queue()
        else:
            self.session = TCPClient(ip_addr, port, send_queue_size=tcp_send_queue_size)
        self.session.start()

    def send(self, data):
        assert isinstance(self.session, TCPClient), f"only TCPClient can call send()! got {self.session}"

        name = data.name if isinstance(data, ApiInfo) else data
        try:
            buffer = io.BytesIO()
            torch.save(data, buffer, pickle_module=dill,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error occurs when saving {name} in bytes, skip it: {e}")
        else:
            data_bytes = buffer.getvalue()
            self.session.cache_data(data_bytes, name=name)

    def recv(self,):
        assert isinstance(self.session, TCPServer), f"only TCPServer can call recv()! got {self.session}"

        try:
            recv_bytes = self.recv_queue.get(block=False)
        except queue.Empty:
            time.sleep(0.1)
            return None

        try:
            data_buffer = io.BytesIO(recv_bytes)
            recv_data = torch.load(
                data_buffer, map_location=self.recv_map_location, pickle_module=dill)
        except Exception as e:
            logger.error(f"Error occurs when loading data from io buffer, skip it: {e}")
            return None
        else:
            name = recv_data.name if hasattr(recv_data, "name") else recv_data
            logger.debug(f"receive data success: {name}")
            return recv_data

    def can_stop(self,):
        return self.session.can_stop()

    def length_of_recv_queue(self):
        assert isinstance(self.session, TCPServer), f"only TCPServer can call length_of_recv_queue()! got {self.session}"
        return self.recv_queue.qsize()
