from abc import ABC, abstractmethod


class BaseFactor(ABC):
    def __init__(self, op_name):
        self.op_name = op_name

    @abstractmethod
    def run(self, data_item):
        pass
