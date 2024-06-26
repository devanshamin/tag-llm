from typing import Optional, Union

import torch
from torch_geometric.data import Data

from tag_llm.config import DatasetName, FeatureType
from tag_llm.data import parser
from tag_llm.data.llm.engine import LlmOfflineEngineArgs, LlmOnlineEngineArgs
from tag_llm.data.lm_encoder import LmEncoder, LmEncoderArgs
from tag_llm.data.parser.base import Parser


class GraphDataset:

    def __init__(
        self,
        dataset_name: DatasetName,
        feature_type: FeatureType,
        lm_encoder_args: LmEncoderArgs,
        llm_online_engine_args: Optional[LlmOnlineEngineArgs] = None,
        llm_offline_engine_args: Optional[LlmOfflineEngineArgs] = None,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = 42,
        cache_dir: str = '.cache'
    ) -> None:

        self.seed = seed
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.llm_online_engine_args = llm_online_engine_args
        self.llm_offline_engine_args = llm_offline_engine_args
        self.cache_dir = cache_dir

        assert llm_online_engine_args or llm_offline_engine_args, \
            'LLM online/offline engine arguments cannot be empty! Please provide either one of them.'

        lm_encoder_args.device = device
        self.lm_encoder = LmEncoder(args=lm_encoder_args)

        self._parser: Optional[Parser] = None
        self._topk = None

    @property
    def dataset(self) -> Data:
        if self._parser is None:
            self.load_dataset()
            self.update_node_features()
        return self._parser.dataset

    @property
    def num_classes(self) -> int:
        return self._parser.graph.n_classes

    @property
    def topk(self) -> int:
        """TopK ranked LLM predictions."""

        if self._topk is None:
            _ = self.dataset
            self._topk = min(self._parser.graph.n_classes, 5)
        return self._topk

    def load_dataset(self) -> None:
        if self.dataset_name == DatasetName.PUBMED:
            cls = parser.PubmedParser
        elif self.dataset_name == DatasetName.OGBN_ARXIV:
            cls = parser.OgbnArxivParser
        else:
            raise ValueError(f'Invalid dataset name "{self.dataset_name}"!')

        self._parser = cls(seed=self.seed, cache_dir=self.cache_dir)
        self._parser.parse()

    def update_node_features(self) -> None:
        """Update original node features with Language Model (LM) features."""

        ftype = self.feature_type
        print(f"Generating node features for feature type '{ftype.name} ({ftype.value})'...")

        if ftype == FeatureType.TITLE_ABSTRACT:
            sentences = [
                f'Title: {article.title}\nAbstract: {article.abstract}'
                for article in self._parser.articles
            ]
            features = self.lm_encoder(sentences)
            features = torch.stack(features)
            self.lm_encoder.save_cache()
        else:
            responses = self._get_llm_responses()

            if ftype == FeatureType.EXPLANATION:
                features = self.lm_encoder(sentences=[resp.reason for resp in responses])
                features = torch.stack(features)
                self.lm_encoder.save_cache()
            else:
                # FeatureType.PREDICTION
                label2id = {label.name: label.label_idx for label in self._parser.class_labels}
                features = torch.zeros((self.dataset.num_nodes, self.topk))
                for i, resp in enumerate(responses):
                    # Convert the predicted labels (which are strings) to their corresponding integer IDs.
                    preds = [label2id[label] for label in resp.label]

                    # Assign the converted predictions to the corresponding row in the features tensor.
                    # `preds` can have fewer elements than `topk`, so we only fill as many elements as we have in `preds`.
                    # We add 1 to each ID because the nn.Embedding layer typically expects non-zero indices to learn embeddings.
                    # Zero can be used to represent padding or a non-existent class.
                    features[i][:len(preds)] = torch.tensor(preds, dtype=torch.long) + 1

                    # Explanation of why we add 1 to the labels:
                    # The OGBN-Arxiv dataset contains LLM predictions where the labels are fixed topk values.
                    # In contrast, the PubMed dataset contains LLM predictions where the labels can be either a single value or multiple values.
                    # During GNN training, the features tensor is passed to an nn.Embedding layer.
                    # If we had topk=3 and preds = [0], initializing the features with zeros would make it difficult to distinguish
                    # between "no prediction" and "prediction of class 0". To denote that the class is present, we increment the value by 1.

        self.dataset.x = features

    def _get_llm_responses(self) -> None:
        if self.llm_online_engine_args:
            from tag_llm.data.llm import online as engine

            args = self.llm_online_engine_args
        else:
            from tag_llm.data.llm import offline as engine

            args = self.llm_offline_engine_args

        if self.dataset_name == DatasetName.PUBMED:
            cls = engine.LlmPubmedResponses
        elif self.dataset_name == DatasetName.OGBN_ARXIV:
            cls = engine.LlmOgbnArxivResponses

        llm = cls(args=args, class_labels=self._parser.class_labels)
        responses = llm.get_responses_from_articles(articles=self._parser.articles)
        return responses
