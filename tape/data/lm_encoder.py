import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Literal

import torch
from tqdm import tqdm

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

        if args.model_library == 'transformers':
            from transformers import AutoTokenizer, AutoModel

            self.model = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=cache_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
        elif args.model_library == 'sentence_transformer':
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(args.model_name_or_path, device=self.device, cache_folder=cache_dir)
        else:
            raise Exception(f'Invalid model library!')
    
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

    def __call__(self, articles: List[str], **kwargs) -> torch.Tensor:

        if self.args.model_library == 'transformers':
            default_kwargs = asdict(self.args.transformers_encoder_args)
            default_kwargs.update(kwargs) # kwargs overrides the default config
            batch_size = kwargs['batch_size']
            embeddings = []
            for step in tqdm(range(0, len(articles), batch_size), total=len(articles)//batch_size, desc='Batches'):
                embeddings.append(self._hf_encoder(articles=articles[step:step + batch_size]))
            embeddings = torch.cat(embeddings)
        elif self.args.model_library == 'sentence_transformer':
            default_kwargs = asdict(self.args.sentence_transformer_encoder_args)
            default_kwargs.update(kwargs) # kwargs overrides the default config
            default_kwargs.pop('convert_to_tensor', None)
            embeddings = self.model.encode(articles, convert_to_tensor=True, **kwargs)
        return embeddings
