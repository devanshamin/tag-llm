from typing import Dict

from tag_llm.data.llm.engine import LlmOfflineEngineArgs
from tag_llm.data.llm.offline.base import LlmOfflineEngine


class LlmOgbnArxivResponses(LlmOfflineEngine):

    def __init__(self, args: LlmOfflineEngineArgs, class_id_to_label: Dict) -> None:
        super().__init__(args)
        self.class_id_to_label = class_id_to_label

    def get_system_prompt(self) -> str:
        topk = 5
        categories = [
            f"{v['label']} // {v['category'].replace('-', ' ').replace(',', '')}"
            for v in self.class_id_to_label.values()
        ]
        kwargs = dict(
            role="You're an experienced computer scientist.",
            categories=categories,
            label_description=f'Contains {topk} arXiv CS sub-categories ordered from most to least likely.',
        )
        prompt = self.load_system_prompt_from_template(**kwargs)
        return prompt
