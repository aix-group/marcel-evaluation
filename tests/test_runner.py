import json
import os
from pathlib import Path

import pytest

from marcel_evaluation import llm, runner
from marcel_evaluation.base import Context, Response
from marcel_evaluation.metrics.non_answer_critic_prompt import ClassifiedAnswer

CONFIG = {"dataset": "dummy", "run_id": "dummy"}


RESPONSES = [
    Response(
        id="1",
        question="Question 1",
        generated_answer=["w1 w2 w3 w4", "w1", "w1"],
        reference_answer="w1 w2",
        sources=["d1", "d2"],
        contexts=[Context(content="text", score=10, url="d1")],
        latency=1,
    ),
    Response(
        id="2",
        question="Question 2",
        generated_answer=["w1 w2", "w1 w2 w3", "w1 w2 w3 w4"],
        reference_answer="w1",
        sources=["d2"],
        contexts=[Context(content="text", score=10, url="d2")],
        latency=1,
    ),
]


@pytest.fixture(autouse=True, scope="module")
def set_env():
    os.environ["WANDB_MODE"] = "disabled"


@pytest.fixture
def dummy_run_path(tmp_path):
    with open(tmp_path / "config.json", "w") as fout:
        json.dump(CONFIG, fout)

    with open(tmp_path / "output.json", "w") as fout:
        json.dump(RESPONSES, fout)

    return tmp_path


@pytest.fixture(autouse=True, scope="function")
def mock_generate(monkeypatch):
    async def _mock_generate(
        messages,
        model,
        response_format,
        n,
        temperature,
        max_tokens,
    ):
        if response_format is ClassifiedAnswer:
            return [ClassifiedAnswer(rationale="", non_answer=0)]

    monkeypatch.setattr(llm, "generate", _mock_generate)


@pytest.mark.asyncio
async def test_runner(dummy_run_path: Path):
    class args:
        run_path: str = str(dummy_run_path)
        model: str = ""
        metrics: str = "GeneratedAnswerLength,ReferenceAnswerLength,NonAnswerCritic"
        force: bool = False

    await runner.main(args)

    files_actual = set([p.name for p in dummy_run_path.iterdir() if p.is_file()])
    files_expected = {
        "wandb.json",
        "config.json",
        "metrics_by_sample.json",
        "metrics.json",
        "metrics.raw.json",
        "output.json",
    }
    assert files_actual == files_expected

    with open(dummy_run_path / "metrics.json") as fin:
        metrics_actual = json.load(fin)

    assert metrics_actual == {
        "GeneratedAnswerLength": 2.5,
        "ReferenceAnswerLength": 1.5,
        "NonAnswerCritic": 1.0,
    }

    with open(dummy_run_path / "metrics.raw.json") as fin:
        metrics_raw_actual = json.load(fin)

    assert metrics_raw_actual == {
        "GeneratedAnswerLength": {
            "score": 2.5,
            "raw": [{"score": 2.0, "raw": [4, 1, 1]}, {"score": 3.0, "raw": [2, 3, 4]}],
        },
        "ReferenceAnswerLength": {"score": 1.5, "raw": [2, 1]},
        "NonAnswerCritic": {
            "score": 1.0,
            "raw": [{"score": 0.0, "raw": [0, 0, 0]}, {"score": 0.0, "raw": [0, 0, 0]}],
        },
    }
