from abc import ABC, abstractmethod


class OnlineAgent(ABC):
    '''A abstract class of Online Session Part.
    Define the interface to transport data including PyTorch Tensors.
    '''
    @abstractmethod
    def send(self):
        pass

    @abstractmethod
    def recv(self):
        pass

    def can_stop(self):
        return False
