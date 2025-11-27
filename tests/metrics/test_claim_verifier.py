import pytest

from marcel_evaluation.metrics import claim_verifier
from marcel_evaluation.metrics.claim_verifier import (
    ClaimEntailmentVerifier,
    Output,
    Verdict,
)


def test_build_prompts():
    texts = ["doc1", "doc2"]
    claims = [["c1", "c2"], ["c3"]]
    verifier = ClaimEntailmentVerifier(model="")
    prompts = verifier.build_prompts(texts, claims)
    assert len(prompts) == 2
    assert "c1" in prompts[0][0]["content"]
    assert "c2" in prompts[0][0]["content"]
    assert "c3" in prompts[1][0]["content"]


@pytest.mark.asyncio
async def test_calculate_entailment(monkeypatch):
    texts = ["doc1", "doc2", "doc3"]
    claims = [["c1", "c2"], ["c3"], ["c4"]]

    async def mock_batch_generate(*args, **kwargs):
        return [
            Output(
                verdicts=[
                    Verdict(statement="c1", reason="r1", verdict=1),
                    Verdict(statement="cc", reason="r2", verdict=0),
                ]
            ),
            Output(
                verdicts=[
                    Verdict(statement="c3", reason="r3", verdict=1),
                ]
            ),
            Output(
                verdicts=[
                    Verdict(statement="c4", reason="r4", verdict=0),
                ]
            ),
        ]

    monkeypatch.setattr(claim_verifier, "batch_generate", mock_batch_generate)

    verifier = ClaimEntailmentVerifier(model="")
    result = await verifier.calculate_entailment(texts, claims)
    assert result == [[1, 0], [1], [0]]
