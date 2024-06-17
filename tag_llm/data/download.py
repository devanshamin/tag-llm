import zipfile
from pathlib import Path
from shutil import copy2

import gdown
import requests


DATASET_TO_URL = {
    'ogbn-arxiv': {
        'original': 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz',
        'llm_responses': 'https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing',
    },
    'ogbn-products': {
        'original': 'dataset/ogbn_products_orig/ogbn-products_subset.csv',
        'llm_responses': 'https://drive.google.com/file/d/1C769tlhd8pT0s7I3vXIEUI-PK7A4BB1p/view?usp=sharing',
    },
    'arxiv_2023': {
        'original': 'https://drive.google.com/file/d/1-s1Hf_2koa1DYp_TQvYetAaivK9YDerv/view?usp=sharing',
        'llm_responses': 'https://www.dropbox.com/scl/fi/cpy9m3mu6jasxr18scsoc/arxiv_2023.zip?rlkey=4wwgw1pgtrl8fo308v7zpyk59&dl=0',
    },
    'Cora': {
        'original': 'https://drive.google.com/file/d/1hxE0OPR7VLEHesr48WisynuoNMhXJbpl/view?usp=share_link',
        'llm_responses': 'https://drive.google.com/file/d/1tSepgcztiNNth4kkSR-jyGkNnN7QDYax/view?usp=sharing',
    },
    'PubMed': {
        'original': 'https://drive.google.com/file/d/1sYZX-jP6H8OkopVa9cp8-KXdEti5ki_W/view?usp=sharing',
        'llm_responses': 'https://drive.google.com/file/d/166waPAjUwu7EWEvMJ0heflfp0-4EvrZS/view?usp=sharing',
    },
}

CACHE_DIR = Path.cwd() / '.cache'
CACHE_DIR.mkdir(exist_ok=True)


def _download(url, destination):

    response = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32_768):
            if chunk:
                f.write(chunk)

def download_and_unzip_file(dataset_name: str, output_dir=CACHE_DIR):

    dtype_to_local_path = {}

    for dtype, url in DATASET_TO_URL[dataset_name].items():
        save_dir = output_dir / dtype
        save_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = save_dir / f'{dataset_name}.zip'
        dir_path = save_dir / f'{dataset_name}'

        if not url.startswith('http'):
            file_path = str(dir_path / Path(url).name)
            copy2(url, file_path)
            dtype_to_local_path[dtype] = file_path
            continue
        
        if not dir_path.exists():
            if 'drive.google.com' in url:
                file_id = url.split('/d/')[1].split('/')[0]
                download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
                gdown.download(download_url, str(zip_file_path), quiet=False)
            elif 'dropbox.com' in url:
                url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
            _download(url, zip_file_path)

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(str(save_dir))
            
            zip_file_path.unlink()
        
        dtype_to_local_path[dtype] = dir_path
    
    return dtype_to_local_path


if __name__ == '__main__':

    for k in DATASET_TO_URL:
        out = download_and_unzip_file(k)
        print(out)
