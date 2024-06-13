import json
from enum import Enum
from pathlib import Path
from typing import Literal, List, Dict, Optional, Generator

from tqdm import tqdm
from litellm import ModelResponse
from pydantic import BaseModel, Field

from tape.data.parser.ogb_arxiv import Article
from tape.data.llm.base import LlmConnector, LlmResponseModel, LlmConnectorArgs


class OgbArxivLlmResponses(LlmConnector):

    def __init__(self, args: LlmConnectorArgs, class_id_to_label: Dict) -> None:
        
        args.dataset_name = 'ogb_arxiv'
        super().__init__(args)
        self.class_id_to_label = class_id_to_label
        self._response_model = None

    def __call__(self, article: Article, **inference_kwargs) -> Optional[LlmResponseModel]:
        
        response = self.inference(
            messages=[
                dict(role='system', content='Which arXiv CS sub-category does this paper belong to?'),
                dict(role='user', content='Title: {}\nAbstract: {}'.format(article.title, article.abstract)),
            ],
            **inference_kwargs
        )
        return response

    def get_response_model(self) -> LlmResponseModel:
        
        # Note: PaperCategory enum (individal class) description doesn't show up 
        # as part of `.model_json_schema()`.
        # This is fine for this case where you have 40 classes and adding their description
        # will result into quite a long prompt impacting accuracy, cost and response time.
        PaperCategory = self._get_paper_category_enum()

        class RankedClass(BaseModel):
            category: PaperCategory # type: ignore
            rank: Literal[1, 2, 3, 4, 5] # topk = 5

        class Classification(LlmResponseModel):
            label: List[RankedClass] = Field(..., description='Provide 5 likely arXiv CS sub-categories ordered from most to least likely.')
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
    
    def llm_responses_reader(self, responses_dir: Path) -> Generator[LlmResponseModel, None, None]:
        
        raise NotImplementedError(
            'Reading LLM responses is not supported for the OGB Arxiv dataset '
            'because the JSON files do not contain comma-separated labels (ranked labels).'
        )

        # llm_model = self.args.inference_args.model.split('/')[-1]
        # files = list(responses_dir.iterdir())
        
        # for child in tqdm(files, total=len(files), desc='Reading LLM responses'):
        #     response = ModelResponse(**json.loads(child.read_text()))
            
        #     if response.model != llm_model:
        #         raise ValueError("Model name doesn't match! Please provide a valid model name.")
            
        #     content = response.choices[0].message.content
        #     ranked_labels, *explanation = content.split('\n\n')
            
        #     response = self.response_model(
        #         label=[dict(category=label.upper().replace('CS', 'cs').strip(' .'), rank=i) for i, label in enumerate(ranked_labels.split(','), start=1)],
        #         reason=''.join(explanation).strip()
        #     )
            
        #     yield response