from typing import Optional, Union

import torch
from torch_geometric.data import Data

from tape.data import parser
from tape.config import DatasetName, FeatureType
from tape.data.lm_encoder import LmEncoder, LmEncoderArgs
from tape.data.llm.engine import LlmOnlineEngineArgs, LlmOfflineEngineArgs


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

        self._parser = None
        self._dataset = None
        self._topk = None
    
    @property
    def dataset(self) -> Data:
        if self._dataset is None:
            self.load_dataset()
            self.update_node_features()
        return self._dataset
    
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
            cls = parser.OgbArxivParser
        else:
            raise ValueError(f'Invalid dataset name "{self.dataset_name}"!')
        
        self._parser = cls(seed=self.seed, cache_dir=self.cache_dir)
        self._parser.parse()
        self._dataset = self._parser.graph.dataset
    
    def _get_llm_responses(self):
        
        graph = self._parser.graph

        if self.llm_online_engine_args:
            from tape.data.llm import online as engine

            args = self.llm_online_engine_args
        else:
            from tape.data.llm import offline as engine

            args = self.llm_offline_engine_args

        if self.dataset_name == 'pubmed':
            cls = engine.LlmPubmedResponses
        elif self.dataset_name == 'ogb_arxiv':
            cls = engine.LlmOgbArxivResponses

        llm = cls(args=args, class_id_to_label=graph.class_id_to_label)
        responses = llm.get_responses_from_articles(articles=graph.articles) 
        return responses

    def update_node_features(self) -> None:
        """Update original node features with Language Model (LM) features."""

        graph = self._parser.graph
        articles = graph.articles
        if self.feature_type == 'title_abstract':
            sentences = [
                f'Title: {article.title}\nAbstract: {article.abstract}'
                for article in articles
            ]
            features = self.lm_encoder(sentences)
        else:
            responses = self._get_llm_responses()
            
            if self.feature_type == 'explanation':
                features = self.lm_encoder(sentences=[resp.reason for resp in responses])
            else:
                # prediction
                label2id = {
                    v['label'] if isinstance(v, dict) else v: k 
                    for k, v in graph.class_id_to_label
                }
                features = torch.zeros((self._dataset.num_nodes, self.topk))
                for i, resp in enumerate(responses):
                    preds = [label2id[label] for label in resp.label]
                    features[i] = torch.tensor(preds, dtype=torch.long)
        
        self.lm_encoder.save_cache()
        self._dataset.x = features