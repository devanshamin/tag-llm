from pathlib import Path
from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import Union, Optional, List, Dict

from pydantic import BaseModel
from dotenv import load_dotenv

from tag_llm.data.parser import Article

load_dotenv()


@dataclass
class LlmOnlineEngineArgs:
    model: str
    max_retries: int = 5
    sampling_kwargs: Optional[Dict] = None # Arguments for OpenAI's `client.chat.completions.create` method
    rate_limit_per_minute: Optional[int] = None # Requests per minute (RPM)
    cache_dir: str = '.cache'

    def __post_init__(self) -> None:
        if self.cache_dir:
            self.cache_dir = str(Path.cwd() / self.cache_dir)


@dataclass
class LlmOfflineEngineArgs(LlmOnlineEngineArgs):
    batch_size: int = 100
    # sampling_kwargs ➜ Arguments for `vllm.EngineArgs`
    engine_kwargs: Optional[Dict] = None # Arguments for `vllm.EngineArgs`


class LlmResponseModel(BaseModel, ABC):
    label: List[str]
    reason: str


class LlmEngine(ABC):
    def __init__(
        self, 
        args: Union[LlmOnlineEngineArgs, LlmOfflineEngineArgs]
    ) -> None:
        self.args = args
    
    @abstractmethod
    def __call__(self) -> Optional[LlmResponseModel]:
        pass

    @abstractmethod
    def get_responses_from_articles(self, articles: List[Article]) -> List[LlmResponseModel]:
        pass