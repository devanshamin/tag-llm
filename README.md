# LLM-Enhanced Text-Attributed Graph Representation Learning

- This repository provides a comprehensive framework designed to leverage large language models (LLMs) for enhancing text-attributed graph (TAG) representation learning. By integrating LLMs, the framework significantly boosts the performance of graph neural networks (GNNs) on various downstream tasks.

- Currently, the repository supports the methodology presented in the paper [Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning](https://arxiv.org/abs/2305.19523).

## Framework Overview

1. **Node Feature Extraction**
   - Prepare prompts containing the article information (title and abstract) for each node.
   - Query an LLM with these prompts to generate a ranked label prediction list and explanation.

2. **Node Feature Encoder**
   - Fine-tune a language model (LM) on a sequence classification task with the article title and abstract as input.

3. **GNN Trainer**
   - Train a GNN model using the following features, with node features updated by the fine-tuned LM encoder:
     1. Title & Abstract (TA)
     2. Prediction (P) - Using a PyTorch `nn.Embedding` layer for top-k ranked features.
     3. Explanation (E)

4. **Model Ensemble**
   - Fuse predictions from the trained GNN models on TA, P, and E by averaging them.

> \[!Note\]
> Fine-tuning an LM is optional and not currently supported. Instead, you can use any open-weight fine-tuned embedding model, significantly reducing time and cost while achieving comparable results.

## Design Choices

Two crucial components of this project were LLM inference and LM inference, each with specific challenges and solutions.

### LLM Inference

#### Challenges
1. **Rate Limits and Cost:**
   - Using provider APIs (OpenAI, Anthropic, Groq, etc.) was straightforward but slow and expensive due to rate limits on requests per minute (RPM) and tokens per minute (TPM).
   
2. **Throughput with Naive Pipelines:**
   - Naive Hugging Face text generation pipeline was slow with open-weight models.

#### Solutions
1. **Online Inference:**
   - **Provider APIs:** Utilized APIs from providers like OpenAI, Anthropic, Groq, etc.
   - **Unified Interface:** Used the [litellm](https://github.com/BerriAI/litellm) package to connect to different LLM providers via a unified interface.
   - **Structured Outputs:** Employed the [instructor](https://github.com/jxnl/instructor) package for structured outputs using Pydantic classes.
   - **Rate Limit Handling:** Implemented exponential backoff retrying and proactive delay by setting the [rate_limit_per_minute](./train_config.yaml) parameter in the configuration.
   - **Durability:** Ensured durability with persistent caching of LLM responses using [diskcache](https://github.com/grantjenks/python-diskcache).

2. **Offline Inference:**
   - **Open-Weights Models:** Used publicly available open-weight models from the Hugging Face hub, opting for mid-sized (7-8 billion parameter) models to balance performance and cost.
   - **Batch Inference:** Maximized throughput by using the [vLLM](https://github.com/vllm-project/vllm) engine for batch inference.
   - **Structured Output Challenges:** Addressed the challenge of getting structured outputs from open-weight models with prompt engineering and a generalizable [prompt template](/tape/data/llm/offline/prompt.jinja), validated with Python code and retried as necessary.

### LM Inference

#### Challenges
1. **Encoding Speed:**
   - Encoding could be slow depending on the size of the model and dataset.

#### Solutions
1. **Model Selection:**
   - Implemented support for models from both [transformers](https://github.com/huggingface/transformers) and [sentence-transformers](https://github.com/UKPLab/sentence-transformers).
   - Opted for <200 million parameter embedding models for faster encoding with decent performance, using the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) as a guide.
   
2. **Caching:**
   - Utilized [safetensors](https://github.com/huggingface/safetensors) for safe storage and distribution of cached embeddings, improving the speed and efficiency of the process.

By addressing these bottlenecks with strategic choices in both online and offline LLM inference and efficient LM inference, the framework ensured enhanced performance and scalability.

## Usage

### Setup the environment
```bash
$ conda create -n tape python=3.10 -y

# Replace the 'cu118' CUDA version according to your system
$ pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
$ pip install torch_geometric
$ pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

# For online LLM inference
$ poetry install
# For offline LLM inference
$ poetry install --extras "llm_offline"
```

### Training

```bash
$ tag_llm_train --config=train_config.yaml
# You can also provide CLI arguments to overwrite values in the `train_config.yaml` file
$ tag_llm_train --help
```

## License

LitGNN is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license. See the [LICENSE](LICENSE) file for details.