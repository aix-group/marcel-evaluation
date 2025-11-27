import numpy as np
import pytest

from marcel_evaluation.base import Context, Response
from marcel_evaluation.metrics.answer_faithfulness import AnswerFaithfulness
from marcel_evaluation.metrics.claim_splitter import ClaimSplitter
from marcel_evaluation.metrics.claim_verifier import ClaimEntailmentVerifier


def test_calculate_score():
    metric = AnswerFaithfulness(model="")

    # each response has a list of claims, 1: entailed, 0: not entailed.
    entailment_labels = [
        [1, 1, 0, 0],  # 2/4
        [0, 0],  # 0/2
        [0, 1],  # 1/2
        [0],  # 0/1
        [1],  # 1/1
        [],  # nan
    ]

    score, scores, entailed, total = metric.calculate_score(entailment_labels)
    assert score == 0.4
    assert scores == [0.5, 0, 0.5, 0, 1, np.nan]
    assert entailed == [2, 0, 1, 0, 1, 0]
    assert total == [4, 2, 2, 1, 1, 0]


def test_is_valid_response():
    r_answer_and_context = {
        "generated_answer": ["text"],
        "contexts": [Context(url="", content="text", score=0)],
    }
    r_no_answer = {
        "generated_answer": [],
        "contexts": [Context(url="", content="text", score=0)],
    }
    r_empty_answer = {
        "generated_answer": [""],
        "contexts": [Context(url="", content="text", score=0)],
    }
    r_no_context = {
        "generated_answer": ["text"],
        "contexts": [],
    }
    r_empty_context = {
        "generated_answer": ["text"],
        "contexts": [Context(url="", content="", score=0)],
    }
    r_no_answer_no_context = {
        "generated_answer": [""],
        "contexts": [],
    }

    metric = AnswerFaithfulness(model="")
    assert metric.is_valid_response(r_answer_and_context)
    assert not metric.is_valid_response(r_no_answer)
    assert not metric.is_valid_response(r_empty_answer)
    assert not metric.is_valid_response(r_no_context)
    assert not metric.is_valid_response(r_empty_context)
    assert not metric.is_valid_response(r_no_answer_no_context)


class MockSplitter(ClaimSplitter):
    async def extract_claims(self, responses: list[Response]) -> list[list[str]]:
        """
        Dummy claim splitter where each whitespace-separated token is a claim.
        """

        return [response["generated_answer"][0].split(" ") for response in responses]


class MockVerifier(ClaimEntailmentVerifier):
    async def calculate_entailment(
        self, texts: list[str], claims_per_text: list[list[str]]
    ) -> list[list[int]]:
        """
        Dummy claim verifier. A claim is entailed by text if the claim literally equals 'entail'.
        """
        return [
            [int(claim == "entail") for claim in claims] for claims in claims_per_text
        ]


@pytest.mark.asyncio
async def test_compute():
    responses = [
        {
            "generated_answer": ["entail entail contradict contradict"],
            "contexts": [Context(url="", content="text", score=0)],
        },
        {
            "generated_answer": ["contradict"],
            "contexts": [Context(url="", content="text", score=0)],
        },
        {
            "generated_answer": ["entail"],
            "contexts": [Context(url="", content="text", score=0)],
        },
        # edge case: no answer and hence no claims
        {
            "generated_answer": [""],
            "contexts": [],
        },
        # edge case: no contexts
        {
            "generated_answer": ["entail"],
            "contexts": [],
        },
    ]

    metric = AnswerFaithfulness(
        model="",
        claim_splitter=MockSplitter(model=""),
        claim_verifier=MockVerifier(model=""),
    )

    result = await metric.compute_async(responses)
    assert result == {
        "score": 0.5,
        "raw": [0.5, 0, 1, np.nan, np.nan],
        "entailed": [2, 0, 1, 0, 0],
        "total": [4, 1, 1, 0, 0],
        "claims": [
            ["entail", "entail", "contradict", "contradict"],
            ["contradict"],
            ["entail"],
            [],
            [],
        ],
    }


@pytest.mark.asyncio
async def test_compute_no_valid_responses():
    responses = [
        {
            "generated_answer": [""],
            "contexts": [],
        }
    ]

    metric = AnswerFaithfulness(
        model="",
        claim_splitter=MockSplitter(model=""),
        claim_verifier=MockVerifier(model=""),
    )

    result = await metric.compute_async(responses)
    assert result == {
        "score": np.nan,
        "raw": [np.nan],
        "entailed": [0],
        "total": [0],
        "claims": [[]],
    }
