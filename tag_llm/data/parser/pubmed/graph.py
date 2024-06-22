import json
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.datasets import Planetoid

from tag_llm.data.parser.base import Article, ClassLabel, Graph


class PubmedGraph(Graph):
    def __init__(self, dir_path: Path, articles_file_path: Path, cache_dir: Path) -> None:
        super().__init__(articles_file_path, cache_dir)
        self.dir_path = dir_path
        self.n_classes = 3
        self.class_labels = [
            ClassLabel(
                label_idx=0,
                name='Experimental Diabetes',
                description='Studies investigating diabetes in controlled experimental settings.',
            ),
            ClassLabel(
                label_idx=1,
                name='Type 1 Diabetes',
                description=(
                    'An autoimmune disease where the body attacks and destroys insulin-producing cells '
                    'in the pancreas.'
                ),
            ),
            ClassLabel(
                label_idx=2,
                name='Type 2 Diabetes',
                description=(
                    "A metabolic disorder characterized by high blood sugar levels due to the body's "
                    'inability to effectively use insulin.'
                ),
            ),
        ]
        self.n_nodes = 19_717
        self.n_features = 500
        self._pubmed_id_to_node_id = {}
        self._node_features: Optional[torch.tensor] = None
        self._node_labels: Optional[torch.tensor] = None
        self._node_feature_to_idx = {}
        self._edge_index: Optional[torch.tensor] = None
        self._adj_matrix: Optional[torch.tensor] = None

    def load(self) -> None:
        self.load_articles()
        self._load_nodes()
        self._load_edges()
        self.load_dataset()

    def load_articles(self):
        print('Loading articles...')
        self.articles = []
        data = json.loads(self.articles_file_path.read_text())
        node_id = 0
        for article in data:
            if (pubmed_id := article.get('PMID')) and (title := article.get('TI')) and (abstract := article.get('AB')):
                self.articles.append(Article(paper_id=pubmed_id, title=title, abstract=abstract))
                self._pubmed_id_to_node_id[pubmed_id] = node_id
                node_id += 1
            else:
                print(f'Ignoring PubMed article with node id "{node_id}" due to missing PMID/Abstract/Title.')

        print(f'No. of PubMed articles with title and abstract: {len(self.articles):,}')
        print(f'Updating no. of nodes from {self.n_nodes:,} to {len(self.articles):,}')
        self.n_nodes = len(self.articles)

    def _load_nodes(self) -> None:
        print('Loading nodes...')
        self._node_features = torch.zeros((self.n_nodes, self.n_features), dtype=torch.float32)
        self._node_labels = torch.empty(self.n_nodes, dtype=torch.long)

        with open(self.dir_path / 'data/Pubmed-Diabetes.NODE.paper.tab', 'r') as node_file:
            node_file.readline() # Ignore header
            node_file.readline() # Ignore header
            k = 0

            for line in node_file.readlines():
                items = line.strip().split('\t')
                pubmed_id = items[0]
                if (node_id := self._pubmed_id_to_node_id.get(pubmed_id)) is None:
                    print(f'Ignoring PubMed article "{pubmed_id}" due to missing PMID/Abstract/Title.')
                    continue

                label = int(items[1].split('=')[-1]) - 1
                self._node_labels[node_id] = label
                features = items[2:-1]
                for feature in features:
                    parts = feature.split('=')
                    fname = parts[0]
                    fvalue = float(parts[1])
                    if fname not in self._node_feature_to_idx:
                        self._node_feature_to_idx[fname] = k
                        k += 1
                    self._node_features[node_id, self._node_feature_to_idx[fname]] = fvalue

    def _load_edges(self) -> None:
        print('Loading edges...')
        edges = []
        self._adj_matrix = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.float32)

        with open(self.dir_path / 'data/Pubmed-Diabetes.DIRECTED.cites.tab', 'r') as edge_file:
            edge_file.readline() # Ignore header
            edge_file.readline() # Ignore header
            for line in edge_file.readlines():
                items = line.strip().split('\t')
                tail = items[1].split(':')[-1]
                head = items[3].split(':')[-1]
                if (
                    (head_node_id := self._pubmed_id_to_node_id.get(head)) is None
                    or (tail_node_id := self._pubmed_id_to_node_id.get(tail)) is None
                ):
                    print(f'Ignoring edge ({head}, {tail}) due to either of the PubMed articles being discarded.')
                    continue

                self._adj_matrix[tail_node_id, head_node_id] = 1.0
                self._adj_matrix[head_node_id, tail_node_id] = 1.0
                if head != tail:
                    edges.append((head_node_id, tail_node_id))
                    edges.append((tail_node_id, head_node_id))

        edges = torch.tensor(edges, dtype=torch.long)
        self._edge_index = torch.unique(edges, dim=0).T

    def load_dataset(self) -> None:
        print('Loading PyG dataset...')
        self.dataset = Planetoid(self.cache_dir, 'PubMed')[0]
        # Replace dataset matrices with the PubMed-Diabetes data,
        # for which we have the original PubMed IDs
        self.dataset.x = self._node_features
        self.dataset.y = self._node_labels
        self.dataset.edge_index = self._edge_index

        # Split dataset nodes into train/valid/test and update the train/valid/test masks
        n_nodes = self.dataset.num_nodes
        node_ids = torch.randperm(n_nodes)
        self.split = {}
        for split_name in ('train', 'valid', 'test'):
            if split_name == 'train':
                subset = slice(0, int(n_nodes * 0.6))
            elif split_name == 'valid':
                subset = slice(int(n_nodes * 0.6), int(n_nodes * 0.8))
            else:
                subset = slice(int(n_nodes * 0.8), n_nodes)

            ids = node_ids[subset].sort()[0]
            setattr(self.dataset, f'{split_name}_id', ids)
            mask = torch.zeros(n_nodes, dtype=bool)
            mask[ids] = True
            setattr(self.dataset, f'{split_name}_mask', mask)
            self.split[split_name] = ids
