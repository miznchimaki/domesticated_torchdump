import io
from twisted.internet import reactor, protocol
from threading import Thread
from queue import Queue

from torchdump.utils import get_logger

from .utils import DELIMITER

logger = get_logger()

class TCPServer:
    def __init__(self, port=-1, recv_queue_size=20) -> None:
        self.port = port
        self.recv_queue = Queue(maxsize=recv_queue_size)
        self.factory = TensorServerFactory(self.recv_queue)

    def start(self):
        self.conn_port = reactor.listenTCP(self.port, self.factory, interface='::')
        logger.info("Start listening...")

        # make sure only call reactor.run once
        if not reactor.running:
            self.reactor_thread = Thread(target=reactor.run, args=(False,), daemon=True)
            self.reactor_thread.start()

    def can_stop(self):
        if self.factory.connection_nums == 0:
            logger.info("Stop listening...")
            if self.conn_port:
                self.conn_port.stopListening()
            if reactor.running:
                reactor.callFromThread(reactor.stop)
                self.reactor_thread.join()
            return True
        return False

    def get_recv_queue(self):
        return self.recv_queue


class TensorServerProtocol(protocol.Protocol):
    def __init__(self, recv_queue) -> None:
        self.buffer = io.BytesIO()
        self.buffer_size = 0
        self.recv_queue = recv_queue

    def connectionMade(self):
        logger.info("Connect with client successed.")
        self.factory.connection_nums += 1

    def connectionLost(self, reason) -> None:
        logger.info(f"Connect with client lost: {reason.getErrorMessage()}")
        self.factory.connection_nums -= 1
        # make sure that all data in buffer have been parsed
        self._try_parse_data()

    def dataReceived(self, data):
        # move pointer to end of buffer
        self.buffer.seek(0, io.SEEK_END)
        self.buffer.write(data)
        self.buffer_size += len(data)

        if DELIMITER in data:
            self._try_parse_data()

    def _try_parse_data(self):
        self.buffer.seek(0)
        recv_bytes = self.buffer.getvalue()
        segments = recv_bytes.split(DELIMITER)
        for api_bytes in segments[:-1]:
            if api_bytes:
                self.recv_queue.put(api_bytes, block=True)
        self.buffer = io.BytesIO(segments[-1])
        self.buffer_size = len(segments[-1])


class TensorServerFactory(protocol.Factory):
    def __init__(self, recv_queue) -> None:
        self.connection_nums = 0
        self.connections = set()
        self.recv_queue = recv_queue

    def buildProtocol(self, addr):
        proto = TensorServerProtocol(self.recv_queue)
        proto.factory = self
        self.connections.add(proto)
        return proto

    def stopFactory(self):
        # close all active connections
        for conn in list(self.connections):
            conn.transport.loseConnection()
        self.connections.clear()
        logger.info("Connect with all clients closed.")
