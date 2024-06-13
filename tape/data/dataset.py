from typing import Literal, Optional, Union

import torch
from tqdm import tqdm
from torch_geometric.data import Data

from tape.data import parser
from tape.data.lm_encoder import LmEncoder, LmEncoderArgs
from tape.data.llm import LlmConnectorArgs, OgbArxivLlmResponses, PubmedLlmResponses


class GraphDataset:

    def __init__(
        self,
        dataset_name: str,
        feature_type: Literal['title_abstract', 'prediction', 'explanation'],
        lm_encoder_args: LmEncoderArgs,
        llm_connector_args: LlmConnectorArgs,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = 42,
        cache_dir: str = '.cache'
    ) -> None:
        
        self.seed = seed
        self.dataset_name = dataset_name.lower()
        self.feature_type = feature_type
        self.cache_dir = cache_dir
        self.llm_connector_args = llm_connector_args

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
        if self.dataset_name == 'pubmed':
            cls = parser.PubmedParser
        elif self.dataset_name == 'ogb_arxiv':
            cls = parser.OgbArxivParser
        else:
            raise ValueError(f'Invalid dataset name "{self.dataset_name}"!')
        
        self._parser = cls(seed=self.seed, cache_dir=self.cache_dir)
        self._parser.parse()
        self._dataset = self._parser.graph.dataset
    
    def update_node_features(self) -> None:
        """Update original node features with Language Model (LM) features."""

        graph = self._parser.graph
        articles = graph.articles
        if self.feature_type == 'title_abstract':
            texts = [
                f'Title: {article.title}\nAbstract: {article.abstract}'
                for article in articles
            ]
            features = self.lm_encoder(texts)
        else:
            if self.dataset_name == 'pubmed':
                cls = PubmedLlmResponses
            elif self.dataset_name == 'ogb_arxiv':
                cls = OgbArxivLlmResponses
            llm = cls(
                args=self.llm_connector_args,
                class_id_to_label=graph.class_id_to_label
            )
            responses = llm.get_responses_from_articles(articles)
            
            if self.feature_type == 'explanation':
                texts = [resp.reason for resp in responses]
                features = self.lm_encoder(texts)
            else:
                # prediction
                label2id = {
                    v['label'] if isinstance(v, dict) else v: k 
                    for k, v in graph.class_id_to_label
                }
                features = torch.zeros((self._dataset.num_nodes, self.topk))
                for i, resp in enumerate(responses):
                    preds = [label2id[label.value] for label in resp.label]
                    features[i] = torch.tensor(preds, dtype=torch.long)
        
        self._dataset.x = features