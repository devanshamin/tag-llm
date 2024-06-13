from abc import abstractmethod, ABC
from pathlib import Path


class Parser(ABC):

    def __init__(self, seed: int = 0, cache_dir: str = '.cache') -> None:
        self.seed = seed
        self.cache_dir = Path.cwd() / cache_dir
    
    @abstractmethod
    def parse(self):
        pass
    
    @abstractmethod
    def download_data(self):
        pass
