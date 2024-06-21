import zipfile
from pathlib import Path
from typing import Tuple

import gdown

from tag_llm.data.parser.base import Parser
from tag_llm.data.parser.pubmed.graph import PubmedGraph


class PubmedParser(Parser):
    """Parser for [PubMed Diabetes](https://linqs.org/datasets/#pubmed-diabetes) dataset."""

    url = 'https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing'

    def __init__(self, seed: int = 0, cache_dir: str = '.cache') -> None:
        super().__init__(seed, cache_dir)
        dir_path, articles_file_path = self.download_data()
        self.graph = PubmedGraph(
            dir_path=dir_path,
            articles_file_path=articles_file_path,
            cache_dir=self.cache_dir
        )

    def parse(self) -> None:
        self.graph.load()

    def download_data(self) -> Tuple[Path, Path]:
        save_dir = self.cache_dir / 'original'
        save_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = save_dir / 'PubMed.zip'
        unzipped_file_path = save_dir / 'PubMed_orig'
        articles_file_path = unzipped_file_path / 'pubmed.json'

        if not articles_file_path.exists():
            file_id = PubmedParser.url.split('/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
            gdown.download(download_url, str(zip_file_path), quiet=False)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(str(save_dir))
            zip_file_path.unlink()

        return unzipped_file_path, articles_file_path
