import io
from time import sleep
from twisted.internet import reactor, protocol
from twisted.protocols.basic import FileSender
from threading import Thread
from queue import Queue, Empty

from torchdump.utils import get_logger

from .utils import DELIMITER

logger = get_logger()

class TCPClient:
    def __init__(
            self,
            ip_addr,
            port=-1,
            send_queue_size=10,
            send_queue_timeout=60
            ) -> None:
        self.ip_addr = ip_addr
        self.port = port
        self.send_queue = Queue(maxsize=send_queue_size)
        self.send_queue_timeout = send_queue_timeout
        self.factory = TensorClientFactory()
        self.sockets = self.factory.protocols

    def start(self):
        reactor.connectTCP(self.ip_addr, self.port, self.factory, timeout=30)
        logger.info("Start client...")

        # make sure only call reactor.run once
        if not reactor.running:
            self.reactor_thread = Thread(target=reactor.run, args=(False,), daemon=True)
            self.reactor_thread.start()

        self.send_thread = Thread(target=self._send_data_loop, daemon=False)
        self.send_thread.start()

    def can_stop(self):
        # disconnect until all data has been sent
        while not self.send_queue.empty():
            sleep(0.01)
        logger.info("Stopping client.")
        try:
            for socket in self.sockets:
                # ensure last data has been sent completely
                while socket.is_sending:
                    sleep(0.01)
                socket.transport.loseConnection()
        except Exception as e:
            logger.warning(f"transport.loseConnection() raises: {e}")
            for socket in self.sockets:
                socket.is_lost = True
        return True

    def cleanup(self):
        # clean up resources before exiting
        try:
            while True:
                self.send_queue.get_nowait()
        except Empty:
            pass

    def cache_data(self, data, name="data"):
        if self.factory.connect_failed:
            self.cleanup()
            raise Exception(f"[online] Error occurs when sending {name} because connection failed.")
        if self.sockets and self.sockets[0].is_lost:
            self.cleanup()
            raise Exception(f"[online] Error occurs when sending {name} because connection lost.")
        try:
            self.send_queue.put(data, block=True, timeout=self.send_queue_timeout)
        except Exception as e:
            logger.error(f"Error occurs or timeout when caching {name} into send_queue, will skip sending it: {e}")

    def _send_data_loop(self):
        while True:
            if self.factory.connect_failed:
                self.cleanup()
                break

            if not self.sockets:
                continue

            # currently, only support one client
            socket = self.sockets[0]
            if socket is None or (not socket.is_connected) or socket.is_sending:
                continue

            if self.send_queue.qsize() > 0:
                if socket.is_lost:
                    self.cleanup()
                    break
                try:
                    #print(f"send_queue size: {self.send_queue.qsize()}")
                    data = self.send_queue.get()
                    socket.send_next_data(data)
                except Exception as e:
                    raise e

            if socket.is_lost:
                self.cleanup()
                break


class TensorClientProtocol(protocol.Protocol):
    def __init__(self) -> None:
        self.is_connected = False
        self.is_lost = False
        self.is_sending = False
        self.buffer = io.BytesIO()

    def connectionMade(self):
        logger.info("Connect with server successed.")
        self.is_connected = True

    def connectionLost(self, reason) -> None:
        logger.info(f"Connect with server lost: {reason.getErrorMessage()}")
        self.is_lost = True

    def dataReceived(self, data):
        # handling data received from server here
        return super().dataReceived(data)

    def send_next_data(self, data):
        self.is_sending = True

        self.buffer.write(data + DELIMITER)
        self.buffer.seek(0)

        # Need to ensure Reactor-related operations(e.g.,beginFileTransfer) in reactor thread,
        # in case undefined behaviour or incorrect dispatch.
        reactor.callFromThread(self._add_sender_to_reactor_call_queue)

    def _add_sender_to_reactor_call_queue(self,):
        file_sender = FileSender()
        # default value: 2**14=16KB, set as 2**20=1MB
        file_sender.CHUNK_SIZE = 2**20
        d = file_sender.beginFileTransfer(self.buffer, self.transport)
        d.addCallback(self._on_transfer_finished)
        d.addErrback(self._on_transfer_failed)
        d.addTimeout(60, reactor)

    def _on_transfer_finished(self, _):
        # clear io buffer
        self.buffer.truncate(0)
        self.buffer.seek(0)
        self.transport.unregisterProducer()
        self.is_sending = False

    def _on_transfer_failed(self, failure):
        logger.error(f"sending data failed or timeout, skip it: {failure}")
        # clear io buffer
        self.buffer.truncate(0)
        self.buffer.seek(0)
        self.transport.unregisterProducer()
        self.is_sending = False


class TensorClientFactory(protocol.ClientFactory):
    def __init__(self) -> None:
        # in order to support multiple connections
        self.protocols = []
        self.connect_failed = False

    def buildProtocol(self, addr):
        proto = TensorClientProtocol()
        self.protocols.append(proto)
        return proto

    def clientConnectionFailed(self, connector, reason):
        self.connect_failed = True
        logger.info(f"Connection Failed: {reason.getErrorMessage()}")

    def clientConnectionLost(self, connector, reason):
        logger.info(f"Connection Lost: {reason.getErrorMessage()}")
