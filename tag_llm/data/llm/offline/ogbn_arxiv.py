from typing import List

from tag_llm.data.llm.engine import LlmOfflineEngineArgs
from tag_llm.data.llm.offline.base import LlmOfflineEngine
from tag_llm.data.parser import ClassLabel


class LlmOgbnArxivResponses(LlmOfflineEngine):

    def __init__(self, args: LlmOfflineEngineArgs, class_labels: List[ClassLabel]) -> None:
        super().__init__(args)
        self.class_labels = class_labels

    def get_system_prompt(self) -> str:
        topk = 5
        categories = [
            f"{label.name} // {label.kwargs['category'].replace('-', ' ').replace(',', '')}"
            for label in self.class_labels
        ]
        kwargs = dict(
            role="You're an experienced computer scientist.",
            categories=categories,
            label_description=f'Contains {topk} arXiv CS sub-categories ordered from most to least likely.',
        )
        prompt = self.load_system_prompt_from_template(**kwargs)
        return prompt
