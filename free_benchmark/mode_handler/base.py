from abc import ABC, abstractmethod


class BaseModeHandler(ABC):
    def __init__(self, op_name, stage) -> None:
        self.op_name = op_name
        self.stage = stage

    @abstractmethod
    def handle(self, data_item, disturb_factor, step, error_save=True):
        pass

    def need_replace_output(self):
        return False
