import json
import time
import functools
from typing import Union
from pathlib import Path

import diskcache
from pydantic import BaseModel
from litellm import ModelResponse

from tape.data.utils import generate_string_hash


CACHE = None

def setup_cache(cache_dir: Union[str, Path]) -> None:
    global CACHE
    CACHE = diskcache.Cache(cache_dir)


def llm_responses_cache(func):
    """Cache a function that returns a Pydantic model."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        assert CACHE is not None, 'Cache is not set! Please call `set_cache(...)` before calling the function.'
        delay = kwargs.pop('delay', 0)
        response_model = kwargs['response_model']
        key = f'{func.__name__}-{make_key(args, kwargs)}'        
        if (cached := CACHE.get(key)) is not None:
            # Deserialize from JSON based on the return type
            usage = ModelResponse(**json.loads(cached))
            data = usage.choices[0].message.tool_calls[0].function.arguments
            response = response_model.model_validate_json(data)
        else:
            time.sleep(delay)
            # Call the function and cache its result
            response, usage = func(*args, **kwargs)
            serialize_usage = usage.model_dump_json()
            CACHE.set(key, serialize_usage)
        return response

    return wrapper


def make_key(args, kwargs):
    data = ''
    convert = lambda v: str(v.model_json_schema() if isinstance(v, BaseModel) else v)
    for arg in args:
        data += convert(arg)
    for k, v in kwargs.items():
        data += k + convert(v)
    input_hash = generate_string_hash(data)
    return input_hash