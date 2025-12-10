# RAG Evaluation

A library to evaluate RAG systems including IR metrics (e.g., MRR, r@k, p@k), classic NLG metrics (ROUGE, BLEU, BERTScore) and LLM-as-a-judge inspired by [`ragas`](https://github.com/explodinggradients/ragas).

## Prepare system outputs

We expect that system outputs are formatted as a list of JSON objects as in the example below. `contexts` are the results of your retrieval system, sorted in descending order of relevance. The `contexts.url` and `sources` fields are used to calculate IR metrics such as Mean Reciprocal Rank (MRR). If your system generates multiple candidate answers, pass a list in `generated_answer`. The sample-level score will be determined by the average of per-answer scores.

<details>
<summary>Data example</summary>

```json
[
  {
    "id": "0",
    "question": "What is the highest peak in Jotunheimen National Park?",
    "generated_answer": [
      "The highest peak in Jotunheimen National Park is Glittertind. It is 2,465 metres (8,087 ft) tall."
    ],
    "contexts": [
      {
        "content": "More than 250 peaks rise above an elevation of 1,900 metres (6,200 ft), including Northern Europe's two highest peaks: Galdh\u00f8piggen at 2,469 metres (8,100 ft), and Glittertind at 2,465 metres (8,087 ft).",
        "url": "website1",
        "score": 7
      },
      {
        "content": "Jotunheimen National Park is part of the larger area Jotunheimen.",
        "url": "website1",
        "score": 6
      }
    ],
    "reference_answer": "Galdh\u00f8piggen at 2,469 metres (8,100 ft).",
    "sources": ["website1"]
  }
]
```

</details>

## Usage in Python

For llm-as-a-judge metrics, provide an OpenAI compatible endpoint:

```sh
export OPENAI_BASE_URL=...
export OPENAI_API_KEY=...
```

All metrics have the same signature. At minimum, a metric returns a dictionary with a `score`. Metrics may also return additional information such as raw scores on each sample. For demonstration, consider a simple "metric" that calculates the length of the generated answer in words.

```py
from marcel_evaluation.utils import load_run
from marcel_evaluation.metrics import GeneratedAnswerLength

responses = load_run('example/output.json')
metric = GeneratedAnswerLength()
metric.compute(responses)
# Output: {'score': 21.333333333333332, 'raw': [20, 11, 17, 31, 26, 23]}
```

You can evaluate all candidate answers as follows:

```py
from marcel_evaluation import MultiGenerationMetricWrapper
from marcel_evaluation.metrics import GeneratedAnswerLength

# For demo purposes this is only passing the required properties for this metric.
responses = [
    {"generated_answer": ['First candidate', 'The second candidate']},
    {"generated_answer": ['Candidate for response 2', 'And another candidate for response 2']}
]

metric = GeneratedAnswerLength()
metric = MultiGenerationMetricWrapper(metric)
metric.compute(responses)
# Output:
# {
#     'score': 3.75,
#     'raw': [
#         {'score': 2.5, 'raw': [2, 3]},
#         {'score': 5.0, 'raw': [4, 6]}
#     ]
# }
```

To determine if a metric needs wrapping, use the `metric.uses_generated_answer` property:

```py
if metric.uses_generated_answer:
    metric = MultiGenerationMetricWrapper(metric)
```

LLM-as-a-judge metrics make use of async-concurrency model. Call the `compute_async` method.

```py
import asyncio
from marcel_evaluation.metrics import NonAnswerCritic

async def compute_metrics():
    metric = NonAnswerCritic(model='google/gemma-3n-E4B-it')
    await metric.compute_async(responses)

asyncio.run(compute_metrics())
```

## Evaluate a single run

A run is a directory with an `output.json` and `config.json` file (see `example`). Use the script below to evaluate a system run with all metrics.

```sh
# For local execution:
pdm run python -m marcel_evaluation.runner \
    --run_path example/ \
    --model neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8 \
    --metrics ContextLength,MeanReciprocalRank,ContextPrecision,ContextRecall \
    --force

# For for slurm-based execution:
sbatch scripts/evaluate.sh \
    --run_path ...
```

## Batch evaluation of multiple runs

Each run will be evaluated by a separate Slurm job. Use the script below to generate a list of tasks, then follow the printed instruction to start the job array.

```sh
./scripts/evaluate_task_list.sh
```

## Metrics

Please find a description of all metrics below, including what parts of the response and gold annotations they use. Notation:

- $x$: Question
- $y$: Generated nswer
- $\hat{y}$: Reference answer
- $C$: Retrieved contexts
- $\hat{C}$: ground truth contexts (ids)

### Statistics

| Name                    | Description                                                  | $x$ | $y$ | $\hat{y}$ | $C$ | $\hat{C}$ |
| ----------------------- | ------------------------------------------------------------ | --- | --- | --------- | --- | --------- |
| Generated answer length | Number of words in generated answer (`nltk.word_tokenize`).  |     | ✔️  |           |     |           |
| Reference answer length | Number of words in reference answer (`nltk.word_tokenize`).  |     |     | ✔️        |     |           |
| Context length          | Number of words in retrieved context (`nltk.word_tokenize`). |     |     |           | ✔️  |           |

### Retriever metrics

| Name                       | Description                                                                                                                                                                                                                                                                                      | $x$ | $y$ | $\hat{y}$ | $C$ | $\hat{C}$ |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- | --- | --------- | --- | --------- |
| Mean reciprocal rank (MRR) | Evaluates the ranking quality of the retrieved contexts with ground-truth relevance judgements. Relevant contexts should be ranked higher.                                                                                                                                                       |     |     |           | ✔️  | ✔️        |
| Context Precision          | How relevant is the retrieved context for arriving at the reference answer? Prompts LLM to judge if a retrieved context is relevant for arriving at the ground-truth answer. Then calculates mean average precision. For a good score, retrieved contexts need to be relevant and highly ranked. | ✔️  |     | ✔️        | ✔️  |           |
| Context Recall             | How complete the context is for generating the ground-truth? Splits reference answer into claims and checks if each claim can be attributed to the retrieved context. The score is given as the fraction of $\text{supported}/\text{total}$ claims.                                              | ✔️  |     | ✔️        | ✔️  |           |
| Precision at Cutoff (p@k)  | The proportion of the top `k` retrieved documents that are relevant.                                                                                                                                                                                                                             |     |     |           | ✔️  | ✔️        |
| Recall at Cutoff (p@k)     | The proportion of all relevant documents that are retrieved in the top `k` results.                                                                                                                                                                                                              |     |     |           | ✔️  | ✔️        |

### Generator metrics

| Name                | Description                                                                                                                                                                                                                                                                                                                       | $x$ | $y$ | $\hat{y}$ | $C$ | $\hat{C}$ |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- | --- | --------- | --- | --------- |
| BLEU                | Classic NLG eval metric via [huggingface evaluation](https://huggingface.co/spaces/evaluate-metric/sacrebleu).                                                                                                                                                                                                                    |     | ✔️  | ✔️        |     |           |
| ROUGE               | Classic NLG eval metric via [huggingface evaluation](https://huggingface.co/spaces/evaluate-metric/rouge).                                                                                                                                                                                                                        |     | ✔️  | ✔️        |     |           |
| BERTScore           | BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. Via [huggingface evaluation](https://huggingface.co/spaces/evaluate-metric/bertscore).                                                                                           |     | ✔️  | ✔️        |     |           |
| Answer Similarity   | Cosine similarity between embeddings of the reference answer an generated answer. Sentence transformer embeddings ([link](https://sbert.net)).                                                                                                                                                                                    |     | ✔️  | ✔️        |     |           |
| Answer Faithfulness | To what extent is the generated answer supported by the retrieved context? Splits the answer into atomic claims and then checks whether each claim is supported by the context. The score is given as the fraction of $\text{supported}/\text{total}$ claims.                                                                     | ✔️  | ✔️  |           | ✔️  |           |
| Answer relevance    | How relevant is the answer to the question? Generates N hypothetical questions based on the generated answer. Then calculates average cosine similarity of the generated questions vs. the actual question. Intuition: if we can reverse engineer the original question from the answer, we assume the answer is highly relevant. | ✔️  | ✔️  |           | ✔️  |           |
| Non-answer critic   | Check if generated_answer is a non-answer (e.g., "i don't know", "i don't have enough information").                                                                                                                                                                                                                              |     | ✔️  |           |     |           |

## Credits

This code is heavily inspired by [`ragas`](github.com/explodinggradients/ragas). Key differences: (1) we include other classical NLG/IR metrics (e.g., MRR or BERTScore); (2) all LLM-as-a-judge metrics use strictly json-guided decoding to improve reliability of metrics; (3) Some prompts were adjusted to support academic information seeking setting.
