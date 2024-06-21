from typing import List

from tag_llm.data.llm.engine import LlmOfflineEngineArgs
from tag_llm.data.llm.offline.base import LlmOfflineEngine
from tag_llm.data.parser import ClassLabel


class LlmPubmedResponses(LlmOfflineEngine):

    def __init__(self, args: LlmOfflineEngineArgs, class_labels: List[ClassLabel]) -> None:
        super().__init__(args)
        self.class_labels = class_labels

    def get_system_prompt(self) -> str:
        kwargs = dict(
            role="You're an experienced medical doctor.",
            categories=[label.name for label in self.class_labels],
            label_description=(
                'Contains the category (or categories if multiple options apply) ordered '
                'from most to least likely.'
            ),
        )
        prompt = self.load_system_prompt_from_template(**kwargs)
        return prompt
