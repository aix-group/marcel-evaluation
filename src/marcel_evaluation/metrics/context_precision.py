from typing import List

import numpy as np

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.llm import batch_generate
from marcel_evaluation.metrics import context_precision_prompt


class ContextPrecision(Metric):
    """How relevant the context is to the ground-truth answer."""

    def __init__(self, model: str, top_k=10):
        """
        Parameters
        ----------
        top_k : int, optional
            Number of contexts to evaluate (taken in order of relevance). By default 10
        """
        self.model = model
        self.temperature = 0
        self.max_tokens = 2048
        self.top_k = top_k

    @property
    def name(self):
        return "ContextPrecision"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def _average_precision(self, labels: List[int]) -> float:
        """Computes average precision over a list of ranked results. Labels should be a list of binary labels, where 1 is relevant, and 0 is irrelevant."""
        denominator = sum(labels) + 1e-10
        numerator = sum(
            [(sum(labels[: i + 1]) / (i + 1)) * labels[i] for i in range(len(labels))]
        )
        score = numerator / denominator
        return score

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        # Step 1: Prompts preparation
        prompts = [
            context_precision_prompt.format_prompt(
                context_precision_prompt.Example(
                    question=response["question"],
                    context=context["content"],
                    answer=response["reference_answer"],
                )
            )
            for response in responses
            if response["contexts"] is not None
            for context in response["contexts"][: self.top_k]
        ]

        # Step 2: Context verification
        verification_results = await batch_generate(
            conversations=prompts,
            model=self.model,
            response_format=context_precision_prompt.Verdict,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )

        # Step 3: Precision computation
        precisions_per_questions = []
        all_labels = []
        current_idx = 0
        for response in responses:
            contexts_cnt = len(response["contexts"][: self.top_k])
            verdicts = verification_results[current_idx : current_idx + contexts_cnt]
            labels = [verdict.verdict for verdict in verdicts]
            all_labels.append(labels)
            precision = self._average_precision(labels)
            precisions_per_questions.append(precision)
            current_idx += contexts_cnt

        agg = np.mean(precisions_per_questions)

        # Step 4: Detailed Output
        return {
            "score": agg,
            "raw": precisions_per_questions,
            "labeled_contexts": all_labels,
        }
