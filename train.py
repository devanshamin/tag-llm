from typing import Literal, Optional

from jsonargparse import ArgumentParser, ActionConfigFile

from tape.data.lm_encoder import LmEncoderArgs
from tape.data.dataset import GraphDataset
from tape.model import NodeClassifierArgs
from tape.trainer.gnn_trainer import GnnTrainerArgs, GnnTrainer
from tape.data.llm.engine import LlmOnlineEngineArgs, LlmOfflineEngineArgs


def get_parser() -> ArgumentParser:

    parser = ArgumentParser(parser_mode='omegaconf') # `omegaconf` for variable interpolation
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--dataset', type=Literal['pubmed', 'ogb_arxiv'])
    parser.add_argument('--feature_type', type=Literal['title_abstract', 'prediction', 'explanation', 'tape'])
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--lm_encoder', type=LmEncoderArgs)
    parser.add_argument('--llm_online_engine', type=Optional[LlmOnlineEngineArgs], default=None)
    parser.add_argument('--llm_offline_engine', type=Optional[LlmOfflineEngineArgs], default=None)
    parser.add_argument('--gnn_model', type=NodeClassifierArgs)
    parser.add_argument('--gnn_trainer', type=GnnTrainerArgs)

    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    print(args)

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
    print(graph_dataset.dataset)
    # trainer = GnnTrainer(
    #     trainer_args=args.gnn_trainer,
    #     graph_dataset=graph_dataset,
    #     model_args=args.gnn_model,
    # )
    # trainer.train()