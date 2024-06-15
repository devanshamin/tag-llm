import hashlib
import warnings
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Literal

import torch
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

warnings.filterwarnings('ignore') # Ignore HuggingFace libraries warnings

@dataclass
class TransformersTokenizerArgs:
    batch_size: int = 32
    truncation: bool = True
    padding: bool = True
    max_length: int = 512

@dataclass
class SentenceTransformerArgs:
    batch_size: int = 32
    show_progress_bar: bool = True
    precision: Literal['float32', 'int8', 'uint8', 'binary', 'ubinary'] = 'float32'

@dataclass
class LmEncoderArgs:
    dataset_name: str
    model_name_or_path: str
    model_library: Literal['transformers', 'sentence_transformer']
    transformers_encoder_args: Optional[TransformersTokenizerArgs] = None
    sentence_transformer_encoder_args: Optional[SentenceTransformerArgs] = None
    device: Optional[str] = None
    cache_dir: str = '.cache'

    def __post_init__(self) -> None:
        if self.model_library == 'transformers' and not self.transformers_encoder_args:
            self.transformers_encoder_args = TransformersTokenizerArgs()
        else:
            if not self.sentence_transformer_encoder_args:
                self.sentence_transformer_encoder_args = SentenceTransformerArgs()


class LmEncoder:
    """Language model article encoder."""

    def __init__(self, args: LmEncoderArgs) -> None:

        self.args = args
        self.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        cache_dir = Path.cwd() / args.cache_dir
        file_name = f'{args.dataset_name}_{args.model_name_or_path.replace("/", "--")}.safetensors'
        self.cached_embd_path = cache_dir / file_name
        self._input_hash_to_embedding = self._load_cache()

        if args.model_library == 'transformers':
            from transformers import AutoTokenizer, AutoModel

            self.model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=cache_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
        elif args.model_library == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(args.model_name_or_path, device=self.device, cache_folder=cache_dir)
        else:
            raise Exception(f'Invalid model library!')
    
    def _load_cache(self):
        input_hash_to_embedding = {}
        if self.cached_embd_path.exists():
            print('Loading cached embeddings...')
            with safe_open(str(self.cached_embd_path), framework="pt", device=self.device) as f:
                for k in f.keys():
                    input_hash_to_embedding[k] = f.get_tensor(k)
        return input_hash_to_embedding

    def save_cache(self) -> None:
        save_file(self._input_hash_to_embedding, str(self.cached_embd_path))
        print(f'Saved embedding file to "{self.cached_embd_path}"')

    @torch.inference_mode()
    def _hf_encoder(self, articles: List[str], **kwargs):
        encoded_articles = self.tokenizer(
            articles, 
            truncation=kwargs.get('truncation', True), 
            padding=kwargs.get('padding', True), 
            return_tensors='pt', 
            max_length=kwargs.get('max_length', 512), 
        ).to(self.device)
        # Encode the queries (use the [CLS] last hidden states as the representations)
        embeddings = self.model(**encoded_articles).last_hidden_state[:, 0, :]
        torch.cuda.empty_cache()
        return embeddings

    def _cache_lookup(self, articles: List[str]):
        embeddings = [
            self._input_hash_to_embedding[k] 
            for k in map(LmEncoder.get_hash, articles) if k in self._input_hash_to_embedding
        ]
        return embeddings

    def __call__(self, articles: List[str], **kwargs) -> torch.Tensor:
        embeddings = self._cache_lookup(articles)
        if len(embeddings) != len(articles): # Recompute if there is an incomplete/partial match
            if self.args.model_library == 'transformers':
                _kwargs = deepcopy(self.args.transformers_encoder_args)
                _kwargs.update(kwargs) # kwargs overrides the default config
                batch_size = _kwargs['batch_size']
                embeddings = []
                for step in tqdm(range(0, len(articles), batch_size), total=len(articles)//batch_size, desc='Batches'):
                    embeddings.append(self._hf_encoder(articles=articles[step:step + batch_size], **_kwargs))
                embeddings = torch.cat(embeddings)
            elif self.args.model_library == 'sentence_transformer':
                _kwargs = deepcopy(self.args.sentence_transformer_encoder_args)
                _kwargs.update(kwargs) # kwargs overrides the default config
                _kwargs.pop('convert_to_tensor', None)
                embeddings = self.model.encode(articles, convert_to_tensor=True, **_kwargs)
            self._input_hash_to_embedding.update(dict(zip(map(LmEncoder.get_hash, articles), embeddings)))
        return embeddings

    @staticmethod
    def get_hash(input_str: str):
        sha256 = hashlib.sha256()
        sha256.update(input_str.encode('utf-8'))
        return sha256.hexdigest()