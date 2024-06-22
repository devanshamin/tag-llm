from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data


@dataclass
class Article:
    paper_id: str
    title: str
    abstract: str

@dataclass
class ClassLabel:
    label_idx: int
    name: str
    description: Optional[str] = None
    kwargs: Optional[Dict] = None


class Parser(ABC):
    def __init__(self, seed: int = 42, cache_dir: str = '.cache') -> None:
        self.seed = seed
        self.cache_dir = Path.cwd() / cache_dir
        self.graph: Optional[Graph] = None

    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def download_data(self):
        pass

    @property
    def dataset(self) -> Data:
        return self.graph.dataset

    @property
    def articles(self) -> List[Article]:
        return self.graph.articles

    @property
    def class_labels(self) -> List[ClassLabel]:
        return self.graph.class_labels


class Graph(ABC):
    def __init__(self, articles_file_path: Path, cache_dir: Path) -> None:
        self.articles_file_path = articles_file_path
        self.cache_dir = Path.cwd() / cache_dir

        self.dataset: Optional[Data] = None
        self.n_classes: Optional[int] = None
        self.class_labels: Optional[List[ClassLabel]] = None
        self.articles: Optional[List[Article]] = None
        # Split containing train/val/test node ids
        self.split: Optional[Dict[str, torch.Tensor]] = None

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def load_articles(self) -> None:
        pass

    @abstractmethod
    def load_dataset(self) -> None:
        pass
