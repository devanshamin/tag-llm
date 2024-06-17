import copy
from typing import Optional
from dataclasses import is_dataclass

import torch
import pandas as pd
import numpy as np
from jsonargparse import ArgumentParser, ActionConfigFile

from tag_llm.config import DatasetName, FeatureType
from tag_llm.data.lm_encoder import LmEncoderArgs
from tag_llm.data.dataset import GraphDataset
from tag_llm.gnn_model import NodeClassifierArgs
from tag_llm.trainer.gnn_trainer import GnnTrainerArgs, GnnTrainer
from tag_llm.data.llm.engine import LlmOnlineEngineArgs, LlmOfflineEngineArgs
from tag_llm.utils import profile_execution


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(parser_mode='omegaconf') # `omegaconf` for variable interpolation
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--dataset', type=DatasetName)
    parser.add_argument('--feature_type', type=FeatureType)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seed_runs', type=Optional[int], default=None)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--lm_encoder', type=LmEncoderArgs)
    parser.add_argument('--llm_online_engine', type=Optional[LlmOnlineEngineArgs], default=None)
    parser.add_argument('--llm_offline_engine', type=Optional[LlmOfflineEngineArgs], default=None)
    parser.add_argument('--gnn_model', type=NodeClassifierArgs)
    parser.add_argument('--gnn_trainer', type=GnnTrainerArgs)
    return parser


def update_feature_type(args, feature_type: FeatureType):
    field_name = 'feature_type'
    args_copy = copy.deepcopy(args)
    for attr, attribute_value in vars(args_copy).items():
        if is_dataclass(attribute_value) and hasattr(attribute_value, field_name):
            field_value = getattr(attribute_value, field_name)
            if isinstance(field_value, FeatureType) and (field_value == FeatureType.TAPE):
                setattr(attribute_value, field_name, feature_type)
        elif attr == field_name:
            if isinstance(attribute_value, FeatureType) and (attribute_value == FeatureType.TAPE):
                setattr(args_copy, attr, feature_type)
    return args_copy


def _train(args):
    graph_dataset = GraphDataset(
        dataset_name=args.dataset,
        feature_type=args.feature_type,
        lm_encoder_args=args.lm_encoder,
        llm_online_engine_args=args.llm_online_engine,
        llm_offline_engine_args=args.llm_offline_engine,
        device=args.device,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    trainer = GnnTrainer(
        trainer_args=args.gnn_trainer,
        graph_dataset=graph_dataset,
        model_args=args.gnn_model,
    )
    test_output = trainer.train()
    return graph_dataset, test_output


@profile_execution
def _run(args):
    if args.feature_type == FeatureType.TAPE:
        logits = []
        pred_rows = []
        for value in ('TA', 'P', 'E'):
            ftype = FeatureType._value2member_map_[value]
            _args = update_feature_type(args, feature_type=ftype)
            graph_dataset, test_output = _train(_args)
            logits.append(test_output.logits)
            ftype_str = f'{ftype.name} ({ftype.value})'
            print(f'[Feature type: {ftype_str}] Test accuracy: {test_output.accuracy:.4f}')
            pred_rows.append(dict(Feature_type=ftype_str, Test_accuracy=test_output.accuracy))
        
        # Fuse predictions of features (TA, P, E) by taking an average  
        logits = torch.stack(logits).mean(dim=0)
        y_true = graph_dataset.dataset.y
        mask = graph_dataset.dataset.test_mask
        test_acc = GnnTrainer.compute_accuracy(logits=logits, y_true=y_true, mask=mask)
        pred_rows.append(
            dict(
                Feature_type=f'{args.feature_type.name} ({args.feature_type.value})',
                Test_accuracy=test_acc
            ),
        )

        print()
        print(pd.DataFrame(pred_rows))
    else:
        # Make sure the feature type used across config is consistent
        _args = update_feature_type(args, feature_type=args.feature_type)
        _, test_output = _train(_args)
        test_acc = test_output.accuracy
        print(
            f'[Feature type: {args.feature_type.name} ({args.feature_type.value})] '
            f'Test accuracy: {test_acc:.4f}'
        )
    return test_acc


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    
    if args.seed_runs is None:
        _run(args)
    else:
        test_accs = []
        for seed in range(args.seed_runs):
            args.seed = seed
            test_acc = _run(args)
            test_accs.append(test_acc)
        ftype_str = f'{args.feature_type.name} ({args.feature_type.value})'
        print(f'[Feature type: {ftype_str}] Test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}')