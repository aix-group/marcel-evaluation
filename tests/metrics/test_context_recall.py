import numpy as np
import pytest

from marcel_evaluation.metrics import context_recall
from marcel_evaluation.metrics.context_recall_prompt import (
    ClassifiedSentence,
    ClassifiedSentencesList,
)


@pytest.mark.asyncio
async def test_compute(monkeypatch):
    metric = context_recall.ContextRecall(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    )

    responses = [
        {
            "question": "Q",
            "reference_answer": "S1. S2.",
            "contexts": [{"content": "C1"}],
        },
        {
            "question": "Q",
            "reference_answer": "S1.",
            "contexts": [{"content": "C1"}],
        },
        {
            "question": "Q",
            "reference_answer": "",
            "contexts": [{"content": "C1"}],
        },
    ]

    async def mock_batch_generate(*args, **kwargs):
        return [
            ClassifiedSentencesList(
                sentences=[
                    ClassifiedSentence(
                        sentence="S1.",
                        reason="rationale",
                        label=1,
                    ),
                    ClassifiedSentence(
                        sentence="S2.",
                        reason="rationale",
                        label=0,
                    ),
                ]
            ),
            ClassifiedSentencesList(
                sentences=[
                    ClassifiedSentence(
                        sentence="S1",
                        reason="rationale",
                        label=1,
                    )
                ]
            ),
            ClassifiedSentencesList(sentences=[]),
        ]

    monkeypatch.setattr(context_recall, "batch_generate", mock_batch_generate)

    result = await metric.compute_async(responses)
    assert result["score"] == (1 / 2 + 1) / 2
    assert result["raw"] == [1 / 2, 1, np.nan]
