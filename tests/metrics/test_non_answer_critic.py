import pytest

from marcel_evaluation.metrics import non_answer_critic
from marcel_evaluation.metrics.non_answer_critic_prompt import ClassifiedAnswer


@pytest.mark.asyncio
async def test_compute(monkeypatch):
    metric = non_answer_critic.NonAnswerCritic(model="mock")
    responses = [
        {"generated_answer": ["Unfortunately, I don't have any knowledge about this."]},
        {"generated_answer": ["The captical of Germany is Berlin"]},
    ]

    async def mock_batch_generate(*args, **kwargs):
        return [
            ClassifiedAnswer(non_answer=1, rationale="rationale"),
            ClassifiedAnswer(non_answer=0, rationale="rationale"),
        ]

    monkeypatch.setattr(non_answer_critic, "batch_generate", mock_batch_generate)

    result = await metric.compute_async(responses)
    assert result["score"] == 0.5
    assert result["raw"] == [1, 0]
    print(result["rationales"])
