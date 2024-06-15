import time
import random
from pathlib import Path
from abc import abstractmethod
from typing import List, Optional

import instructor
from tqdm import tqdm
from litellm import completion

from tape.config import DatasetName
from tape.data.parser import Article
from tape.data.llm.online.cache import setup_cache, llm_responses_cache
from tape.data.llm.engine import LlmEngine, LlmOnlineEngineArgs, LlmResponseModel


class LlmOnlineEngine(LlmEngine):
    
    def __init__(self, args: LlmOnlineEngineArgs, dataset_name: DatasetName) -> None:
        super().__init__(args)
        self.client = instructor.from_litellm(completion)
        setup_cache(cache_dir=Path(args.cache_dir) / f'tape_llm_responses/{dataset_name.value}')
        self._response_model = None
    
    @abstractmethod
    def get_response_model(self) -> LlmResponseModel:
        pass

    @property
    def response_model(self) -> LlmResponseModel:
        if self._response_model is None:
            self._response_model = self.get_response_model()
        return self._response_model

    def __call__(self, article: Article, max_retries: int = 3) -> Optional[LlmResponseModel]:
        
        messages = [
            dict(role='system', content=self.system_message),
            dict(role='user', content='Title: {}\nAbstract: {}'.format(article.title, article.abstract))
        ]
        max_retries = self.args.max_retries or max_retries
        response = None
        retries = 0
        backoff_time = 1  # Initial backoff time in seconds

        while retries < max_retries:
            try:
                response = self._get_response(
                    model=self.args.model,
                    messages=messages,
                    response_model=self.response_model,
                    **self.args.sampling_kwargs
                )
                break
            except Exception as e:
                wait_time = backoff_time * (2 ** retries) + random.uniform(0, 1) # Exponential backoff with jitter
                time.sleep(wait_time)
                retries += 1
                print(f"Retry {retries}/{max_retries} after exception: {e}. Waiting {wait_time:.2f} seconds.")

        if retries == max_retries and response is None:
            print("Max retries reached. Failed to get a successful response.")

        return response

    @llm_responses_cache
    def _get_response(self, **kwargs):
        return self.client.chat.completions.create_with_completion(**kwargs)
    
    def get_responses_from_articles(self, articles: List[Article]) -> List[LlmResponseModel]:
        responses = []
        for article in tqdm(articles, total=len(articles), desc='Fetching LLM responses'):
            if not (response := self(article)):
                raise ValueError('LLM response cannot be empty!')
            response.label = response.label.value # Convert Enum to str
            responses.append(response)
        return responses