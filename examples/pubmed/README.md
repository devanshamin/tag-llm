# [PubMed-Diabetes](https://linqs.org/datasets/#pubmed-diabetes) Dataset

## Training

```bash
$ tag_llm_train --config=examples/pubmed/train_config.yaml --seed_runs=4
```

- The [train_config.yaml](./train_config.yaml) utilizes the online LLM engine with the model [huggingface/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Predictions generated by this model for the PubMed dataset have been uploaded to [Hugging Face](https://huggingface.co/datasets/devanshamin/PubMedDiabetes-LLM-Predictions), which will be downloaded and used instead of calling the LLM during training.
- This optimization significantly accelerates the training process and demonstrates end-to-end training with tape.

## Results

- Instead of fine-tuning an LM on the PubMed dataset, the [train_config.yaml](./train_config.yaml) uses a general-purpose embedding model [avsolatorio/GIST-Embedding-v0](https://huggingface.co/avsolatorio/GIST-Embedding-v0).
- With LLM predictions, you can expect the following run time and accuracy when training the GNN for the `PubMed` dataset using the feature type `TAPE`:

```text
When the LM embeddings cache for the dataset is empty,
Feature_type        Test_accuracy
TITLE_ABSTRACT (TA)      0.908722
PREDICTION (P)           0.889959
EXPLANATION (E)          0.914807
TAPE (TAPE)              0.946501
Run time: 11 minutes and 14.59 seconds

When the LM embeddings cache for the dataset is present,
Feature_type        Test_accuracy
TITLE_ABSTRACT (TA)      0.915061
PREDICTION (P)           0.889452
EXPLANATION (E)          0.923174
TAPE (TAPE)              0.952333
Run time: 1 minute and 0.31 seconds
```

In summary,

| | tag-llm | Author Implementation - [TAPE](https://github.com/XiaoxinHe/TAPE) |
| -- | -- | -- |
| LLM  | `meta-llama/Meta-Llama-3-8B-Instruct` | `openai/gpt-3.5-turbo-0301` |
| LM fine-tuning | ✖ | ✔ |
| GNN layer | `SAGEConv` | `SAGEConv` |
| GNN hparams | `layers=4, hidden_dim=64, dropout=0.1` | `layers=3, hidden_dim=256, dropout=0.5` |
| Seed runs | 4 | 4 |
| Accuracy | `0.9573 ± 0.0032` | `0.9618 ± 0.0053` |
