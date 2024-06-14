from enum import Enum
from typing import List, Dict

from pydantic import Field

from tape.data.llm.online.base import LlmOnlineEngine
from tape.data.llm.engine import LlmOnlineEngineArgs, LlmResponseModel


class LlmPubmedResponses(LlmOnlineEngine):

    def __init__(self, args: LlmOnlineEngineArgs, class_id_to_label: Dict) -> None:
        super().__init__(args=args, dataset_name='pubmed')
        self.class_id_to_label = class_id_to_label
        self.system_message = (
            'Classify a scientific publication (containing title and abstract) '
            'into provided categories.'
        )

    def get_response_model(self) -> LlmResponseModel:
        
        class PaperCategory(str, Enum):
            EXPERIMENTAL_DIABETES = 'Experimental Diabetes'
            TYPE_1_DIABETES = 'Type 1 Diabetes'
            TYPE_2_DIABETES = 'Type 2 Diabetes'

        # class RankedClass(BaseModel):
        #     category: PaperCategory
        #     rank: Literal[1, 2, 3]

        class Classification(LlmResponseModel):
            label: List[PaperCategory] = Field(
                ..., 
                description=(
                    'Provide the most likely category (or categories if multiple options apply) ordered '
                    'from most to least likely.'
                ),
                examples=[['Experimental Diabetes'], ['Type 1 Diabetes', 'Experimental Diabetes']]
            )
            reason: str = Field(
                ..., 
                description=(
                    'Give a detailed explanation with quotes from the abstract explaining why '
                    'the paper is related to the chosen label based on the ranking.'
                ),
                examples=[
                    # Example containing single paper category ➜ Type 2 Diabetes
                    # (
                    #     'The paper specifically states that the study involved 31 subjects with type 2 diabetes '
                    #     'who were randomly assigned to pioglitazone or metformin for 4 months. The entire study '
                    #     'is focused on comparing the effects of these two drugs on hepatic and extra-hepatic '
                    #     'insulin action in people with type 2 diabetes. There is no mention of Type 1 diabetes or '
                    #     'experimentally induced diabetes in the text.'
                    # ),
                    # Example containing multiple paper categories ➜ Type 1 Diabetes & Experimental Diabetes
                    (
                        'Type 1 diabetes is present in the abstract as the study was conducted on cardiac mitochondria from type-I diabetic rats. '
                        'Experimentally induced diabetes is also present in the abstract as the study involved inducing diabetes in rats and '
                        'comparing the mitochondrial function of these rats to control rats.'
                    ),
                ]
            )
        
        return Classification
