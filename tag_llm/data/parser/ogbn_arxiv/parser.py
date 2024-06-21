from pathlib import Path

import requests

from tag_llm.data.parser.base import Parser
from tag_llm.data.parser.ogbn_arxiv.graph import OgbnArxivGraph


class OgbnArxivParser(Parser):
    """Parser for [OGB arXiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) dataset."""

    url = 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'

    def __init__(self, seed: int = 0, cache_dir: str = '.cache') -> None:
        super().__init__(seed, cache_dir)
        articles_file_path = self.download_data()
        self.graph = OgbnArxivGraph(articles_file_path=articles_file_path, cache_dir=self.cache_dir)

    def parse(self) -> None:
        self.graph.load()

    def download_data(self) -> Path:
        save_dir = self.cache_dir / 'original/ogbn-arxiv'
        save_dir.mkdir(exist_ok=True, parents=True)
        articles_file_path = save_dir / OgbnArxivParser.url.split('/')[-1]

        if not articles_file_path.exists():
            response = requests.get(OgbnArxivParser.url, stream=True)
            with open(articles_file_path, 'wb') as f:
                for chunk in response.iter_content(32_768):
                    if chunk:
                        f.write(chunk)

        return articles_file_path
