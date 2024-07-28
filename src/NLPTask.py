from abc import ABC, abstractmethod 

class NLPTask(ABC):
    def __init__(self):
        super().__init__()
        self._dataset = None
        self._preprocessor = None
        self._encoder = None
        self._transformer = None
        self.model = None
        self.tester = None
