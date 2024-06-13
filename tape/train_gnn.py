from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.seed import seed_everything

from tape.model import NodeClassifier

seed_everything(42)
