import pytest

from marcel_evaluation.metrics import claim_splitter
from marcel_evaluation.metrics.claim_splitter import ClaimSplitter, Output


def test_sentence_split():
    splitter = ClaimSplitter(model="")
    assert splitter.split_sentences("S1. S2. S3.") == ["S1.", "S2.", "S3."]


def test_build_prompts():
    responses = [
        {
            "question": "Q1",
            "generated_answer": ["S1. S2. S3."],
        },
        {
            "question": "Q2",
            "generated_answer": ["S4. S5."],
        },
    ]

    splitter = ClaimSplitter(model="")
    prompts, response_indices = splitter.build_prompts(responses)
    assert len(prompts) == 5
    assert response_indices == [0, 0, 0, 1, 1]


def test_group_claims_by_response():
    claims_per_sentence = [
        ["S1C1", "S1C2", "S1C3"],
        ["S2C1"],
        ["S3C1", "S3C2"],
        ["S4C1"],
        ["S5C1", "S5C2", "S5C3"],
        [],
    ]
    response_indices = [0, 0, 0, 1, 1, 2]
    n_responses = 3

    splitter = ClaimSplitter(model="")
    grouped = splitter.group_claims_by_response(
        claims_per_sentence, response_indices, n_responses
    )
    assert len(grouped) == 3
    assert grouped == [
        ["S1C1", "S1C2", "S1C3", "S2C1", "S3C1", "S3C2"],
        ["S4C1", "S5C1", "S5C2", "S5C3"],
        [],
    ]


@pytest.mark.asyncio
async def test_extract_claims(monkeypatch):
    splitter = ClaimSplitter(model="")
    responses = [
        {
            "question": "Q1",
            "generated_answer": ["S1. S2."],
        },
        {
            "question": "Q2",
            "generated_answer": ["S3."],
        },
    ]

    async def mock_batch_generate(*args, **kwargs):
        return [
            Output(simpler_statements=["S1C1", "S1C2", "S1C3"]),
            Output(simpler_statements=["S2C1"]),
            Output(simpler_statements=["S3C1"]),
        ]

    monkeypatch.setattr(claim_splitter, "batch_generate", mock_batch_generate)

    result = await splitter.extract_claims(responses)
    assert result == [
        ["S1C1", "S1C2", "S1C3", "S2C1"],
        ["S3C1"],
    ]
