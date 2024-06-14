from tape.data.llm.offline.base import LlmOfflineEngine


# TODO: Convert to Jinja template to work with all datasets
system_prompt = """You're a experienced medical doctor. Given an article with title and abstract, classify the article into the following categories:
1. Experimental Diabetes
2. Type 1 Diabetes
3. Type 2 Diabetes

Return the output in a JSON format as following:
```json
{
    "label": ["Type 2 Diabetes"], # The most likely category (or categories if multiple options apply) ordered from most to least likely.
    "reason": "..." # A detailed explanation with quotes from the article explaining why the article is related to the chosen label based on the ranking.
}
```
Do not return anything else except the JSON string.
"""


class LlmPubmedResponses(LlmOfflineEngine):

    def get_system_prompt(self) -> str:
        return system_prompt