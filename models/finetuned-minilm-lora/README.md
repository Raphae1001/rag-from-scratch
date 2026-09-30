---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:400
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Security - First Steps Recap
  sentences:
  - Let's first just use the code and see how it works, and then we'll come back to
    understand what's happening.
  - So, in just 3 or 4 extra lines, you already have some primitive form of security.
  - Import and use `BackgroundTasks` with parameters in *path operation functions*
    and dependencies to add background tasks.
- source_sentence: Request Forms and Files Recap
  sentences:
  - Use `File` and `Form` together when you need to receive data and files in the
    same request.
  - Here's an example of how an HTTPS API could look, step by step, paying attention
    mainly to the ideas important for developers.
  - 'Now, if you check the docs, they will show all the additional metadata:


    <img src="/img/tutorial/metadata/image02.png">'
- source_sentence: Request Files Recap
  sentences:
  - Similar to making sure your application is run on startup, you probably also want
    to make sure it is **restarted** after failures.
  - Use `File`, `bytes`, and `UploadFile` to declare files to be uploaded in the request,
    sent as form data.
  - You can see the available versions (e.g. to check what is the current latest)
    in the [Release Notes](../release-notes.md).
- source_sentence: Templates Install dependencies
  sentences:
  - 'Add `jinja2` to your project:


    <div class="termy">


    ```console

    $ uv add jinja2


    ---> 100%

    ```


    </div>'
  - If you are using containers (e.g. Docker, Kubernetes), then there are two main
    approaches you can use.
  - 'Then you can write a template at `templates/item.html` with, for example:


    ```jinja hl_lines="7"

    {!../../docs_src/templates/templates/item.html!}

    ```'
- source_sentence: applications Accessing the app instance
  sentences:
  - Next we will see how to add dependencies to the whole `FastAPI` application, so
    that they apply to each *path operation*.
  - Where a `request` is available (i.e. endpoints and middleware), the app is available
    on `request.app`.
  - If you don't care about any of these terms and you just need to add security with
    authentication based on username and password *right now*, skip to the next chapters.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'applications Accessing the app instance',
    'Where a `request` is available (i.e. endpoints and middleware), the app is available on `request.app`.',
    'Next we will see how to add dependencies to the whole `FastAPI` application, so that they apply to each *path operation*.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5213, 0.1660],
#         [0.5213, 1.0000, 0.1589],
#         [0.1660, 0.1589, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 400 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 100 samples:
  |          | anchor                                                                            | positive                                                                             |
  |:---------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                            | string                                                                               |
  | modality | text                                                                              | text                                                                                 |
  | details  | <ul><li>min: 4 tokens</li><li>mean: 13.35 tokens</li><li>max: 30 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 176.12 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | anchor                                                     | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  |:-----------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Server-Sent Events (SSE) Raw Data</code>             | <code>If you need to send data **without** JSON encoding, use `raw_data` instead of `data`.<br><br>This is useful for sending pre-formatted text, log lines, or special <dfn title="A value used to indicate a special condition or state">"sentinel"</dfn> values like `[DONE]`.<br><br><br>```py<br>from collections.abc import AsyncIterable<br><br>from fastapi import FastAPI<br>from fastapi.sse import EventSourceResponse, ServerSentEvent<br><br>app = FastAPI()<br><br><br>@app.get("/logs/stream", response_class=EventSourceResponse)<br>async def stream_logs() -> AsyncIterable[ServerSentEvent]:<br>    logs = [<br>        "2025-01-01 INFO  Application started",<br>        "2025-01-01 DEBUG Connected to database",<br>        "2025-01-01 WARN  High memory usage detected",<br>    ]<br>    for log_line in logs:<br>        yield ServerSentEvent(raw_data=log_line)<br>```<br><br><br>/// note<br><br>`data` and `raw_data` are mutually exclusive. You can only set one of them on each `ServerSentEvent`.<br><br>///</code>                                                                                     |
  | <code>Async Tests Other Asynchronous Function Calls</code> | <code>As the testing function is now asynchronous, you can now also call (and `await`) other `async` functions apart from sending requests to your FastAPI application in your tests, exactly as you would call them anywhere else in your code.<br><br>/// tip<br><br>If you encounter a `RuntimeError: Task attached to a different loop` when integrating asynchronous function calls in your tests (e.g. when using [MongoDB's MotorClient](https://stackoverflow.com/questions/41584243/runtimeerror-task-attached-to-a-different-loop)), remember to instantiate objects that need an event loop only within async functions, e.g. an `@app.on_event("startup")` callback.<br><br>///</code>                                                                                                                                                                                                                                                                                                                                                                                                                                        |
  | <code>alias `AliasPath` and `AliasChoices`</code>          | <code>??? api "API Documentation"<br><br>    [`pydantic.aliases.AliasPath`][pydantic.aliases.AliasPath]<br><br>    [`pydantic.aliases.AliasChoices`][pydantic.aliases.AliasChoices]<br><br><br>Pydantic provides two special types for convenience when using `validation_alias`: `AliasPath` and `AliasChoices`.<br><br>The `AliasPath` is used to specify a path to a field using aliases. For example:<br><br>```python {lint="skip"}<br>from pydantic import BaseModel, Field, AliasPath<br><br><br>class User(BaseModel):<br>    first_name: str = Field(validation_alias=AliasPath('names', 0))<br>    last_name: str = Field(validation_alias=AliasPath('names', 1))<br>    address: str = Field(validation_alias=AliasPath('contact', 'address'))<br><br>user = User.model_validate({  # (1)!<br>    'names': ['John', 'Doe'],<br>    'contact': {'address': '221B Baker Street'}<br>})<br>print(user)<br>#> first_name='John' last_name='Doe' address='221B Baker Street'<br>```<br><br>1. We are using [`model_validate()`][pydantic.BaseModel.model_validate] to validate a dictionary using the field aliases.<br> ...</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `learning_rate`: 0.0002
- `warmup_steps`: 0.1

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 0.0002
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.1
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.3846 | 5    | 0.7580        |
| 0.7692 | 10   | 0.6957        |
| 1.1538 | 15   | 0.7114        |
| 1.5385 | 20   | 0.5846        |
| 1.9231 | 25   | 0.6859        |
| 2.3077 | 30   | 0.6293        |
| 2.6923 | 35   | 0.5703        |


### Training Time
- **Training**: 18.5 seconds

### Framework Versions
- Python: 3.13.9
- Sentence Transformers: 6.1.0
- Transformers: 5.17.0
- PyTorch: 2.14.0
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->