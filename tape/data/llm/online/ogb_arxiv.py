from enum import Enum
from typing import List, Dict

from pydantic import Field

from tape.data.llm.online.base import LlmOnlineEngine
from tape.data.llm.engine import LlmOnlineEngineArgs, LlmResponseModel


class LlmOgbArxivResponses(LlmOnlineEngine):

    def __init__(self, args: LlmOnlineEngineArgs, class_id_to_label: Dict) -> None:        
        super().__init__(args=args, dataset_name='ogb_arxiv')
        self.class_id_to_label = class_id_to_label
        self.system_message = 'Which arXiv CS sub-category does this paper belong to?'

    def get_response_model(self) -> LlmResponseModel:
        
        # Note: PaperCategory enum (individal class) description doesn't show up 
        # as part of `.model_json_schema()`.
        # This is fine for this case where you have 40 classes and adding their description
        # will result into quite a long prompt impacting accuracy, cost and response time.
        PaperCategory = self._get_paper_category_enum()
        
        # class RankedClass(BaseModel):
        #     category: PaperCategory # type: ignore
        #     rank: Literal[1, 2, 3, 4, 5] # topk = 5

        class Classification(LlmResponseModel):
            label: List[PaperCategory] = Field( # type: ignore
                ..., 
                description='Provide 5 likely arXiv CS sub-categories ordered from most to least likely.'
            ) 
            reason: str = Field(
                ..., 
                description=(
                    'Give a detailed explanation explaining why the paper is related to the chosen label based on the ranking.'
                ),
                examples=[
                    (
                        'The paper is about a new dataset for scene text detection and recognition, which is a topic related to computer vision (cs.CV).'
                        'The paper also mentions the use of deep learning techniques such as DeconvNet, which falls under the sub-category of artificial '
                        'intelligence (cs.AI). The dataset is annotated and involves text recognition, which could also fall under the sub-categories of '
                        'information retrieval (cs.IR) and natural language processing (cs.CL). Finally, the paper discusses the effectiveness of different solutions, '
                        'which could be evaluated using machine learning techniques, falling under the sub-category of machine learning (cs.LG).'
                    ),
                ]
            )

        return Classification

    def _get_paper_category_enum(self) -> Enum:
        """Create Enum class on the fly based on the number of classes."""

        enum_members = {
            item['label'].replace('.', '_').upper(): item['label'] 
            for item in self.class_id_to_label.values()
        }
        PaperCategory = Enum('PaperCategory', enum_members)
        for item in self.class_id_to_label.values():
            label = item['label'].replace('.', '_').upper()
            category = item['category']
            description = item['description']
            enum_value = PaperCategory[label]
            enum_value.description = f'{category} - {description}'
        
        return PaperCategory
