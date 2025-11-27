from typing import List

import numpy as np

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.llm import batch_generate
from marcel_evaluation.metrics import context_recall_prompt


class ContextRecall(Metric):
    """How complete the context is for generating the ground-truth."""

    def __init__(self, model: str, top_k=10):
        """
        Parameters
        ----------
        top_k : int, optional
            Number of contexts to evaluate (taken in order of relevance). By default 10
        """
        self.temperature = 0
        self.max_tokens = 4096
        self.model = model
        self.top_k = top_k

    @property
    def name(self):
        return "ContextRecall"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def format_contexts(self, contexts):
        # concatenates all contexts and escapes special characters including newlines
        text = "\n".join([context["content"] for context in contexts])
        return repr(text)

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        # Step 1: Prompts preparation
        prompts = [
            context_recall_prompt.format_prompt(
                context_recall_prompt.Example(
                    question=response["question"],
                    context=self.format_contexts(response["contexts"][: self.top_k]),
                    answer=response["reference_answer"],
                )
            )
            for response in responses
        ]

        # Step 2: Statements classification
        classification_results = await batch_generate(
            conversations=prompts,
            model=self.model,
            response_format=context_recall_prompt.ClassifiedSentencesList,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )

        # Step 3: Recall computation
        all_sentences = [result.sentences for result in classification_results]
        total = [len(result.sentences) for result in classification_results]
        attributed = [
            sum(sent.label == 1 for sent in result.sentences)
            for result in classification_results
        ]
        scores = [a / t if t > 0 else np.nan for a, t in zip(attributed, total)]
        agg = np.nanmean(scores)

        return {
            "score": agg,
            "raw": scores,
            "total": total,
            "attributed": attributed,
            "sentences": all_sentences,
        }
