import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from tag_llm.data.dataset import GraphDataset
from tag_llm.gnn_model import NodeClassifier, NodeClassifierArgs


@dataclass
class GnnTrainerArgs:
    epochs: int
    lr: float
    weight_decay: float = 0.0
    early_stopping_patience: int = 50
    batch_size: Optional[int] = None # Mini-batch training
    num_neighbors: Optional[int] = None # Mini-batch training
    num_workers: Optional[int] = None # Mini-batch training
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.is_mini_batch_training = self.batch_size is not None
        if self.is_mini_batch_training and not self.num_neighbors:
            print('`gnn_trainer.num_neighbors` was not provided. Using the default value of 10.')
            self.num_neighbors = 10
            self.num_workers = self.num_workers or os.cpu_count()

@dataclass
class GnnTrainerOutput:
    loss: float
    accuracy: float
    logits: torch.Tensor

class TrainingStage(str, Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class GnnTrainer:
    def __init__(self, trainer_args: GnnTrainerArgs, graph_dataset: GraphDataset, model_args: NodeClassifierArgs) -> None:
        self.trainer_args = trainer_args
        self.model_args = model_args
        self.dataset: Data = graph_dataset.dataset
        self.split_idx = graph_dataset.split_idx
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

    def train(self) -> GnnTrainerOutput:
        patience = self.trainer_args.early_stopping_patience
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        dataloaders = None

        for epoch in range(1, self.trainer_args.epochs + 1):
            if self.trainer_args.is_mini_batch_training:
                if dataloaders is None:
                    dataloaders = self._get_dataloaders()
                train_output = self._train_eval_mini_batch(epoch, dataloaders['train'], TrainingStage.TRAIN)
                valid_output = self._train_eval_mini_batch(epoch, dataloaders['valid'], TrainingStage.VALID)
            else:
                train_output = self._train_eval_full_batch(TrainingStage.TRAIN)
                valid_output = self._train_eval_full_batch(TrainingStage.VALID)
            print(
                f'Epoch: {epoch:03d} | Train loss: {train_output.loss:.4f}, '
                f'Valid loss: {valid_output.loss:.4f}, Train accuracy: {train_output.accuracy:.4f}, '
                f'Valid accuracy: {valid_output.accuracy:.4f}'
            )
            if valid_output.loss < best_val_loss:
                best_val_loss = valid_output.loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping on epoch {epoch} due to no improvement in validation loss for {patience} epochs.')
                break

        if self.trainer_args.is_mini_batch_training:
            test_output = self._train_eval_mini_batch(epoch, dataloaders['test'], TrainingStage.TEST)
        else:
            test_output = self._train_eval_full_batch(TrainingStage.TEST)
        return test_output

    def _get_dataloaders(self):
        config = self.trainer_args
        num_neighbors = [config.num_neighbors] * self.model_args.num_layers
        persistent_workers = config.num_workers > 0
        train_dataloader = NeighborLoader(
            data=self.dataset,
            num_neighbors=num_neighbors,
            input_nodes=self.split_idx['train'],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
        )
        valid_dataloader = NeighborLoader(
            data=self.dataset,
            num_neighbors=num_neighbors,
            input_nodes=self.split_idx['valid'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
        )
        test_dataloader = NeighborLoader(
            data=self.dataset,
            num_neighbors=num_neighbors,
            input_nodes=self.split_idx['test'],
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=persistent_workers,
        )
        return dict(train=train_dataloader, valid=valid_dataloader, test=test_dataloader)

    def _train_eval_full_batch(self, stage: TrainingStage):
        if stage == TrainingStage.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        data = self.dataset.to(self.device)
        mask = getattr(data, f'{stage.value}_mask', None)
        assert mask, (
            'Missing `*_mask` attributes from the dataset! `train_mask`, '
            '`valid_mask` and `test_mask` are required for full-batch training.'
        )
        if stage == TrainingStage.TRAIN:
            self.optimizer.zero_grad()
            logits = self.model(data.x, data.edge_index)
            loss = self.criterion(logits[mask], data.y[mask].flatten())
            loss.backward()
            self.optimizer.step()
        else:
            with torch.inference_mode():
                logits = self.model(data.x, data.edge_index)
            loss = self.criterion(logits[mask], data.y[mask].flatten())

        accuracy = GnnTrainer.compute_accuracy(logits, data.y, mask)
        return GnnTrainerOutput(loss=float(loss), accuracy=accuracy, logits=logits)

    def _train_eval_mini_batch(self, epoch: int, dataloader: NeighborLoader, stage: TrainingStage):
        if stage == TrainingStage.TRAIN:
            self.model.train()
        else:
            self.model.eval()

        total_loss = total_correct = total_samples = 0
        num_batches = len(dataloader)
        batch_logits = []
        for batch in tqdm(dataloader, total=num_batches, desc=f'{stage.value.capitalize()}ing epoch {epoch}'):
            batch = batch.to(self.device)
            y = batch.y[:batch.batch_size].flatten()
            if stage == TrainingStage.TRAIN:
                self.optimizer.zero_grad()
                logits = self.model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.inference_mode():
                    logits = self.model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = self.criterion(logits, y)

            total_loss += float(loss)
            total_correct += GnnTrainer.compute_accuracy(logits, y)
            total_samples += y.shape[0]
            batch_logits.append(logits)

        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / total_samples
        logits = torch.cat(batch_logits) # full-batch logits
        return GnnTrainerOutput(loss=avg_loss, accuracy=avg_accuracy, logits=logits)

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None):
        y_pred = logits.argmax(dim=-1)
        if mask is None:
            return int((y_pred == y_true).sum())
        correct = y_pred[mask] == y_true[mask]
        return int(correct.sum()) / int(mask.sum())
