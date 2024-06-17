from abc import abstractmethod, ABC
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Article:
    paper_id: str
    title: str
    abstract: str


class Parser(ABC):

    def __init__(self, seed: int = 42, cache_dir: str = '.cache') -> None:
        self.seed = seed
        self.cache_dir = Path.cwd() / cache_dir
    
    @abstractmethod
    def parse(self):
        pass
    
    @abstractmethod
    def download_data(self):
        pass
