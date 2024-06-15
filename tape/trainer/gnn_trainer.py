from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
from torch_geometric.data import Data

from tape.gnn_model import NodeClassifierArgs, NodeClassifier
from tape.data.dataset import GraphDataset


@dataclass
class GnnTrainerArgs:
    epochs: int
    lr: float
    weight_decay: Optional[float] = 0.0
    device: Optional[str] = None


class GnnTrainer:

    def __init__(self, trainer_args: GnnTrainerArgs, graph_dataset: GraphDataset, model_args: NodeClassifierArgs) -> None:
        
        self.trainer_args = trainer_args
        self.dataset: Data = graph_dataset.dataset
        self.device = trainer_args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        use_predictions = graph_dataset.feature_type == 'prediction'
        if use_predictions:
            # The node features will be the `topk` classes    ➜ Shape: (num_nodes, topk)
            # It will get passed to an embedding lookup layer ➜ Shape: (num_nodes, topk, hidden_dim)
            # And the last two dims will be flattened         ➜ Shape: (num_nodes, topk * hidden_dim)
            model_args.in_channels = graph_dataset.topk * model_args.hidden_channels
        else:
            model_args.in_channels = self.dataset.num_node_features
        model_args.out_channels = graph_dataset.num_classes
        model_args.use_predictions = use_predictions
        self.model = NodeClassifier(model_args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=trainer_args.lr, weight_decay=trainer_args.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self) -> None:
        
        for epoch in range(1, self.trainer_args.epochs + 1):
            train_loss, train_acc = self._train_eval(self.dataset, stage='train')
            val_loss, val_acc = self._train_eval(self.dataset, stage='val')
            print(
                f'Epoch: {epoch:03d} | Train loss: {train_loss:.4f}, '
                f'Val loss: {val_loss:.4f}, Train accuracy: {train_acc:.4f}, '
                f'Val accuracy: {val_acc:.4f}'
            )

        test_loss, test_acc = self._train_eval(self.dataset, stage='test')
        print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}')
    
    def _train_eval(self, data: Data, stage: Literal['train', 'val', 'test']) -> Tuple[float, float]:
        
        if stage == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        data = data.to(self.device)
        mask = getattr(data, f'{stage}_mask')
        if stage == 'train':
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[mask], data.y[mask].flatten())
            loss.backward()
            self.optimizer.step()
        else:
            with torch.inference_mode():
                out = self.model(data.x, data.edge_index)
            loss = self.criterion(out[mask], data.y[mask].flatten())
        
        accuracy = GnnTrainer.compute_accuracy(out, data.y, mask)
        return float(loss), accuracy

    @staticmethod
    def compute_accuracy(out: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        
        y_pred = out.argmax(dim=1)
        correct = y_pred[mask] == y_true[mask]
        return int(correct.sum()) / int(mask.sum())
