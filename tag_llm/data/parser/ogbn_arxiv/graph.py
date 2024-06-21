import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from tag_llm.data.parser.base import Article, ClassLabel, Graph


class OgbnArxivGraph(Graph):
    def __init__(self, articles_file_path: Path, cache_dir: Path) -> None:
        super().__init__(articles_file_path, cache_dir)
        self.n_classes = 40
        self.n_nodes = 169_343
        self.n_features = 128

    def load(self) -> None:
        self.load_dataset()
        self.load_articles()
        self._load_class_label_mapping()

    def load_dataset(self) -> None:
        print('Loading OGB dataset...')

        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.cache_dir, transform=T.ToSparseTensor())
        self.split = dataset.get_idx_split()

        data = dataset[0]

        train_mask = torch.zeros(data.num_nodes).bool()
        train_mask[self.split['train']] = True
        data.train_mask = train_mask

        val_mask = torch.zeros(data.num_nodes).bool()
        val_mask[self.split['valid']] = True
        data.val_mask = val_mask

        test_mask = torch.zeros(data.num_nodes).bool()
        test_mask[self.split['test']] = True
        data.test_mask = test_mask

        data.edge_index = data.adj_t.to_symmetric()

        self.dataset = data

    def load_articles(self) -> None:
        mapping_df = pd.read_csv(
            self.cache_dir / 'ogbn_arxiv/mapping/nodeidx2paperid.csv.gz',
            skiprows=1,
            names=['node_idx', 'paper_id'],
            compression='gzip'
        )
        title_abstract_df = pd.read_table(
            self.articles_file_path,
            header=None,
            names=['paper_id', 'title', 'abstract'],
            compression='gzip'
        )
        df = mapping_df.astype(dict(paper_id=str)).join(title_abstract_df.set_index('paper_id'), on='paper_id')
        self.articles = []
        for row in df.itertuples(index=False):
            self.articles.append(Article(paper_id=row.paper_id, title=row.title, abstract=row.abstract))

    def _load_class_label_mapping(self) -> None:
        mapping_df = pd.read_csv(
            self.cache_dir / 'ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz',
            skiprows=1,
            names=['label_id', 'label'],
            compression='gzip'
        )
        categories = OgbnArxivGraph.fetch_arxiv_category_taxonomy()
        df = pd.DataFrame(categories)
        self.class_labels = []
        for row in mapping_df.itertuples(index=False):
            label = row.label.replace('arxiv cs ', 'cs.').strip().upper().replace('CS', 'cs')
            data = df[df.label == label].iloc[0].to_dict()
            class_label = ClassLabel(
                label_idx=row.label_id,
                name=data['label'],
                description=data['description'],
                kwargs=dict(category=data['category']),
            )
            self.class_labels.append(class_label)

    @staticmethod
    def fetch_arxiv_category_taxonomy(category: str = 'cs') -> List[Dict[str, str]]:
        text = requests.get('https://r.jina.ai/https://arxiv.org/category_taxonomy').text
        sections = re.split(r'#### ', text)[1:]
        data_list = []
        for section in sections:
            match = re.match(rf'({category}\.\w+) \(([^)]+)\)\n\n', section)
            if match:
                label = match.group(1)
                category_name = match.group(2)
                description = section[match.end():].strip()
                data_list.append({
                    'label': label,
                    'category': category_name,
                    'description': description
                })
        return data_list
