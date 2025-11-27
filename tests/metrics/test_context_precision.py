import pytest
from pytest import approx

from marcel_evaluation.metrics import context_precision
from marcel_evaluation.metrics.context_precision_prompt import Verdict


@pytest.mark.asyncio
async def test_compute(monkeypatch):
    metric = context_precision.ContextPrecision(model=None)

    responses = [
        {
            "question": "Q",
            "reference_answer": "S1. S2.",
            "contexts": [{"content": "C1"}, {"content": "C2"}, {"content": "C3"}],
        },
        {
            "question": "Q",
            "reference_answer": "S1.",
            "contexts": [{"content": "C1"}, {"content": "C1"}],
        },
        {
            "question": "Q",
            "reference_answer": "",
            "contexts": [{"content": "C1"}],
        },
    ]

    async def mock_batch_generate(*args, **kwargs):
        # flat list of verdicts for all contexts
        return [
            # first response
            Verdict(reason="r", verdict=1),
            Verdict(reason="r", verdict=0),
            Verdict(reason="r", verdict=1),
            # second response
            Verdict(reason="r", verdict=0),
            Verdict(reason="r", verdict=0),
            # third response
            Verdict(reason="r", verdict=1),
        ]

    monkeypatch.setattr(context_precision, "batch_generate", mock_batch_generate)

    ap = metric._average_precision
    expected_raw = [ap([1, 0, 1]), ap([0, 0]), ap([1])]
    expected = sum(expected_raw) / 3

    result = await metric.compute_async(responses)
    assert result["score"] == expected
    assert result["raw"] == expected_raw


def test_verification_2_contexts():
    metric = context_precision.ContextPrecision(model=None)
    f = metric._average_precision
    # fmt: off
    assert f([1]) == approx(1, rel=1e-7)
    assert f([0]) == approx(0, rel=1e-7)
    assert f([0, 0]) == 0  # 0
    assert f([0, 1]) == approx(0.5, rel=1e-7)  # (0/1 * 0 + 1/2 * 1) / 1 = 0.5
    assert f([1, 0]) == approx(1, rel=1e-7)  # (1/1 * 1 + 1/2 * 0) / 1 = 1
    assert f([1, 1]) == approx(1, rel=1e-7)  # (1/1 * 1 + 2/2 * 1) / 2 = 1
    assert f([0, 0, 0]) == 0
    assert f([0, 0, 1]) == approx(0.3333333, rel=1e-7)  # (0/1 * 0 + 0/2 * 0 + 1/3 * 1) / 1 = 0.3333333
    assert f([0, 1, 0]) == approx(0.5, rel=1e-7)  # (0/1 * 0 + 1/2 * 1 + 1/3 * 0) / 1 = 0.5
    assert f([0, 1, 1]) == approx(0.5833333, rel=1e-7)  # (0/1 * 0 + 1/2 * 1 + 2/3 * 1) / 2 = 0.5833333
    assert f([1, 0, 0]) == approx(1.0, rel=1e-7)  # (1/1 * 1 + 1/2 * 0 + 1/3 * 0) / 1 = 1.0
    assert f([1, 0, 1]) == approx(0.8333333, rel=1e-7)  # (1/1 * 1 + 1/2 * 0 + 2/3 * 1) / 2 = 0.8333333
    assert f([1, 1, 0]) == approx(1.0, rel=1e-7)  # (1/1 * 1 + 2/2 * 1 + 2/3 * 0) / 2 = 1.0
    assert f([1, 1, 1]) == approx(1.0, rel=1e-7)  # (1/1 * 1 + 2/2 * 1 + 3/3 * 1) / 3 = 1.0
    # fmt: on
