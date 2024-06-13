from typing import Literal
from jsonargparse import ArgumentParser, ActionConfigFile

from tape.data.llm import LlmConnectorArgs
from tape.data.lm_encoder import LmEncoderArgs
from tape.data.dataset import GraphDataset
from tape.model import NodeClassifierArgs
from tape.trainer.gnn_trainer import GnnTrainerArgs, GnnTrainer


def get_parser() -> ArgumentParser:

    parser = ArgumentParser(parser_mode='omegaconf') # `omegaconf` for variable interpolation
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_argument('--dataset', type=Literal['pubmed', 'ogb_arxiv'])
    parser.add_argument('--feature_type', type=Literal['title_abstract', 'prediction', 'explanation', 'tape'])
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--lm_encoder', type=LmEncoderArgs)
    parser.add_argument('--llm_connector', type=LlmConnectorArgs)
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
        llm_connector_args=args.llm_connector,
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