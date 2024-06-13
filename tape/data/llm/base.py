import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import abstractmethod, ABC
from typing import List, Dict, Optional, Generator

import instructor
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm
from litellm import completion

from tape.data.parser.ogb_arxiv import Article
from tape.data.llm.cache import setup_cache, llm_responses_cache

load_dotenv()


@dataclass
class LlmInferenceArgs:
    model: str
    max_retries: int = 5
    max_tokens: int = 2048
    # Passed to the `client.chat.completions.create` method of `litellm`
    kwargs: Optional[Dict] = None 

@dataclass
class LlmConnectorArgs:
    inference_args: LlmInferenceArgs
    dataset_name: Optional[str] = None
    cache_dir: str = '.cache'
    
class LlmResponseModel(BaseModel, ABC):
    label: str
    reason: str


class LlmConnector(ABC):

    def __init__(self, args: LlmConnectorArgs) -> None:

        self.args = args
        self.client = instructor.from_litellm(completion)

        dataset = args.dataset_name.lower()
        cache_dir = Path.cwd() / args.cache_dir / f'tape_llm_responses/{dataset}'
        setup_cache(cache_dir)
        self._response_model = None

    @property
    def response_model(self) -> LlmResponseModel:
        if self._response_model is None:
            self._response_model = self.get_response_model()
        return self._response_model
    
    @abstractmethod
    def get_response_model(self) -> LlmResponseModel:
        pass

    @abstractmethod
    def __call__(self) -> Optional[LlmResponseModel]:
        pass

    @abstractmethod
    def llm_responses_reader(self, responses_dir: Path) -> Generator[LlmResponseModel, None, None]:
        pass

    def inference(self, messages: List[Dict], **kwargs) -> Optional[LlmResponseModel]:
        default_kwargs = asdict(self.args.inference_args)
        default_kwargs['kwargs'] = default_kwargs['kwargs'] or {}
        default_kwargs.update(**default_kwargs.pop('kwargs')) # Flatten: Dict[Dict] -> Dict
        default_kwargs.update(kwargs) # kwargs overrides the default config
        input_kwargs = dict(
            messages=messages,
            response_model=self.response_model,
            **default_kwargs
        )

        max_retries = default_kwargs.get('max_retries', 5)
        response = None
        retries = 0
        backoff_time = 1  # Initial backoff time in seconds

        while retries < max_retries:
            try:
                response = self._get_response(**input_kwargs)
                break  # Break if successful
            except Exception as e:
                wait_time = backoff_time * (2 ** retries) + random.uniform(0, 1)  # Exponential backoff with jitter
                time.sleep(wait_time)
                retries += 1
                print(f"Retry {retries}/{max_retries} after exception: {e}. Waiting {wait_time:.2f} seconds.")

        if retries == max_retries and response is None:
            print("Max retries reached. Failed to get a successful response.")

        return response

    @llm_responses_cache
    def _get_response(self, **kwargs):
        return self.client.chat.completions.create_with_completion(**kwargs)

    def get_responses_from_articles(
        self, 
        articles: List[Article],
        responses_dir: Optional[Path] = None, 
        **inference_kwargs
    ) -> List[LlmResponseModel]:
        """Fetch LLM responses for the articles. 

        LLM responses are loaded from the JSON files if all of the following
        conditions are met:
        - If `responses_dir` is provided and the class implements `llm_responses_reader` method.
        - The model name in the `LlmInferenceArgs` matches the model name found in the JSON files.
        - The message content contains comma separated labels (ranked labels).
        
        Args:
            articles: List of articles.
            responses_dir: LLM responses directory containing the author provided responses
              from OpenAI model.
        """

        responses = []

        if responses_dir is not None:
            for response in self.llm_responses_reader(responses_dir):
                responses.append(response)
        else:
            for article in tqdm(articles, total=len(articles), desc='Fetching LLM responses'):
                response = self(article, **inference_kwargs)
                assert response is not None, 'LLM response cannot be empty!'
                responses.append(response)
        
        return responses