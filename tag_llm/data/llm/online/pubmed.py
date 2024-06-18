from typing import Dict, List

from datasets import load_dataset

from tag_llm.config import DatasetName
from tag_llm.data.llm.engine import LlmOnlineEngineArgs, LlmResponseModel
from tag_llm.data.llm.online.base import LlmOnlineEngine
from tag_llm.data.parser import Article


class LlmPubmedResponses(LlmOnlineEngine):

    def __init__(self, args: LlmOnlineEngineArgs, class_id_to_label: Dict) -> None:
        super().__init__(args=args, dataset_name=DatasetName.PUBMED)
        self.class_id_to_label = class_id_to_label
        self.system_message = (
            'Classify a scientific publication (containing title and abstract) '
            'into provided categories.'
        )

    def get_response_model(self) -> LlmResponseModel:
        class_labels = {v['label']: v['label'] for v in self.class_id_to_label.values()}
        labels_list = list(class_labels.values())
        kwargs = dict(
            class_labels=class_labels,
            label_description=(
                'Provide the most likely category (or categories if multiple options apply) ordered '
                'from most to least likely.'
            ),
            label_examples=[labels_list[:1], labels_list[:2]],
            reason_examples=[
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
                    'Type 1 diabetes is present in the abstract as the study was conducted on cardiac '
                    'mitochondria from type-I diabetic rats. Experimentally induced diabetes is also '
                    'present in the abstract as the study involved inducing diabetes in rats and '
                    'comparing the mitochondrial function of these rats to control rats.'
                ),
            ]
        )
        return self.load_response_model_from_template(**kwargs)

    def get_responses_from_articles(self, articles: List[Article]) -> List[LlmResponseModel]:
        if self.args.model == 'huggingface/meta-llama/Meta-Llama-3-8B-Instruct':
            dataset = load_dataset(
                "devanshamin/PubMedDiabetes-LLM-Predictions",
                cache_dir=self.args.cache_dir,
                split='train'
            )
            responses = []
            for sample in dataset:
                response = self.response_model(
                    label=sample['predicted_ranked_labels'].split('; '),
                    reason=sample['explanation']
                )
                responses.append(response)
        else:
            responses = super().get_responses_from_articles(articles)
        return responses
