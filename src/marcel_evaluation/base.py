from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np


class Context(TypedDict):
    url: str
    content: str
    score: float


class Response(TypedDict):
    id: str
    question: str
    reference_answer: str
    sources: list[str]
    contexts: list[Context]
    generated_answer: list[str]
    latency: float


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def uses_generated_answer(self) -> bool:
        pass

    @abstractmethod
    def compute(self, responses: list[Response]) -> dict:
        pass

    @abstractmethod
    async def compute_async(self, responses: list[Response]) -> dict:
        pass


class MultiGenerationMetricWrapper(Metric):
    def __init__(self, metric: Metric):
        self.metric = metric

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: list[Response]) -> dict:
        runs = self._normalize_runs(responses)
        scores_by_run = [self.metric.compute(run) for run in runs]
        return self._aggregate_scores(scores_by_run)

    async def compute_async(self, responses: list[Response]) -> dict:
        runs = self._normalize_runs(responses)
        scores_by_run = [await self.metric.compute_async(run) for run in runs]
        return self._aggregate_scores(scores_by_run)

    def _normalize_runs(self, responses: list[Response]):
        n_generations = len(responses[0]["generated_answer"])
        runs = []
        for i in range(n_generations):
            single_gen_responses = [
                Response(
                    {**response, "generated_answer": [response["generated_answer"][i]]}
                )
                for response in responses
            ]
            runs.append(single_gen_responses)
        return runs

    def _aggregate_scores(self, runs):
        """
        Assume we have 3 inputs each with 2 generations. That makes two "runs" that were evaluated.

        Input example:

            [
                {"score": 3.0, "raw": [5, 2, 2]},  # first run
                {"score": 4.0, "raw": [6, 3, 3]},  # second run
            ]

        We now calculate two values:
        - average score for all runs
        - average score for each sample

        Output example:
            {
                "score": 3.5,  # (3+4)/2
                "raw": [
                    {"score": 5.5, "raw": [5, 6]},
                    {"score": 2.5, "raw": [2, 3]},
                    {"score": 2.5, "raw": [2, 3]},
                ]
            }
        """
        result = {}

        # Macro average score for all runs
        result["score"] = np.mean([run["score"] for run in runs])

        # Average score for each sample
        raw_scores = [run["raw"] for run in runs if "raw" in run]  # List[List[float]]
        if raw_scores:
            raw_scores = np.array(raw_scores).T  # shape: [n_responses, n_runs]
            avg_scores = np.nanmean(raw_scores, axis=1)  # shape: [n_responses]
            scores_by_response = [
                {"score": avg, "raw": list(raw)}
                for avg, raw in zip(avg_scores, raw_scores)
            ]
            result["raw"] = scores_by_response

        return result
