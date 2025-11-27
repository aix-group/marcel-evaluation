from typing import List

import numpy as np

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.metrics.claim_splitter import ClaimSplitter
from marcel_evaluation.metrics.claim_verifier import ClaimEntailmentVerifier


class AnswerFaithfulness(Metric):
    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: int = 8192,
        claim_splitter=None,
        claim_verifier=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.splitter = claim_splitter or ClaimSplitter(model=model)
        self.verifier = claim_verifier or ClaimEntailmentVerifier(model=model)

    @property
    def name(self):
        return "AnswerFaithfulness"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def is_valid_response(self, response: Response):
        has_answer = any(response["generated_answer"])
        has_contexts = any(c["content"] for c in response["contexts"])
        return has_answer and has_contexts

    def concatenate_context(self, responses: list[Response]) -> list[str]:
        return [
            " ".join(context["content"] for context in response["contexts"])
            for response in responses
        ]

    def calculate_score(
        self, responses_claims: list[list[int]]
    ) -> tuple[float, list[float], list[int], list[int]]:
        entailed = [sum(claims) for claims in responses_claims]
        total = [len(claims) for claims in responses_claims]
        scores = [e / t if t > 0 else np.nan for e, t in zip(entailed, total)]
        score = np.nanmean(scores)
        return float(score), scores, entailed, total

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        score = np.nan
        scores = [np.nan] * len(responses)
        entailed = [0] * len(responses)
        total = [0] * len(responses)
        claims = [[] for _ in range(len(responses))]

        # Filter valid responses
        valid_indices = [
            i
            for i, response in enumerate(responses)
            if self.is_valid_response(response)
        ]
        valid_responses = [responses[i] for i in valid_indices]

        if not valid_responses:
            return {
                "score": score,
                "raw": scores,
                "entailed": entailed,
                "total": total,
                "claims": claims,
            }

        texts = self.concatenate_context(valid_responses)
        claims_valid = await self.splitter.extract_claims(valid_responses)
        claim_labels = await self.verifier.calculate_entailment(texts, claims_valid)
        score, scores_v, entailed_v, total_v = self.calculate_score(claim_labels)

        for j, i in enumerate(valid_indices):
            scores[i] = scores_v[j]
            entailed[i] = entailed_v[j]
            total[i] = total_v[j]
            claims[i] = claims_valid[j]

        return {
            "score": score,
            "raw": scores,
            "entailed": entailed,
            "total": total,
            "claims": claims,
        }
