import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from marcel_evaluation.base import Context, Response
from marcel_evaluation.metrics import MeanReciprocalRank
from marcel_evaluation.metrics.classic import (
    ROUGE,
    BERTScore,
    ContextLength,
    CorpusBLEU,
    GeneratedAnswerLength,
    NonAnswerCriticClassic,
    PrecisionAtCutoff,
    RecallAtCutoff,
    ReferenceAnswerLength,
)


@pytest.fixture(scope="module", autouse=True)
def download_models():
    import nltk

    nltk.download("punkt")


def test_generated_answer_length():
    responses = [
        {"generated_answer": ["This is a test"]},
        {"generated_answer": ["Second response"]},
    ]
    metric = GeneratedAnswerLength()
    assert metric.compute(responses) == {
        "score": 3.0,
        "raw": [4, 2],
    }


def test_reference_answer_length():
    responses = [
        {"reference_answer": "This is the reference answer"},
        {"reference_answer": "This one has five words"},
        {"reference_answer": "Short answer"},
    ]
    metric = ReferenceAnswerLength()
    assert metric.compute(responses) == {"score": 4.0, "raw": [5, 5, 2]}


def test_context_length():
    responses = [
        {
            "contexts": [
                Context(url="", content="word", score=1),
                Context(url="", content="another context", score=1),
                Context(url="", content="word word", score=1),
            ]
        },
        {
            "contexts": [
                Context(url="", content="this one is long", score=1),
                Context(url="", content="this one is long too", score=1),
                Context(url="", content="and last context", score=1),
            ]
        },
    ]
    metric = ContextLength()
    assert metric.compute(responses) == {"score": (5 + 12) / 2, "raw": [5, 12]}


def test_corpus_bleu():
    responses = [
        {
            "reference_answer": "The dog bit the man.",
            "generated_answer": ["The dog bit the man."],
        },
        {
            "reference_answer": "It was not unexpected.",
            "generated_answer": ["It wasn't surprising."],
        },
        {
            "reference_answer": "The man bit him first.",
            "generated_answer": ["The man had just bitten him."],
        },
    ]
    metric = CorpusBLEU()
    result = metric.compute(responses)
    assert "raw" not in result
    assert round(result["score"], 4) == 0.4507


def test_rouge():
    responses = [
        {
            "reference_answer": "The quick brown fox jumps over the lazy dog",
            "generated_answer": ["The quick brown dog jumps on the log."],
        },
        {
            "reference_answer": "It was not unexpected.",
            "generated_answer": ["It wasn't surprising."],
        },
    ]
    metric = ROUGE(rouge_type="rouge1")
    result = metric.compute(responses)
    assert_almost_equal(result["score"], 0.4779, decimal=4)
    assert_almost_equal(result["raw"], [0.7058, 0.25], decimal=4)


def test_bertscore():
    responses = [
        {
            "reference_answer": "The quick brown fox jumps over the lazy dog",
            "generated_answer": ["The quick brown dog jumps on the log."],
        },
        {
            "reference_answer": "It was not unexpected.",
            "generated_answer": ["It wasn't surprising."],
        },
    ]
    metric = BERTScore(lang="en", rescale_with_baseline=True)
    result = metric.compute(responses)
    assert_almost_equal(result["score"], 0.7554, decimal=4)
    assert_almost_equal(result["raw"], [0.6905, 0.8202], decimal=4)
    assert_almost_equal(result["precision"], [0.7175, 0.8199], decimal=4)
    assert_almost_equal(result["recall"], [0.6627, 0.8199], decimal=4)


def test_mrr_empty_qrels():
    # fmt: off
    # (query_id, sources, list of retrieved contexts: (url, score))
    responses = [
        ("q1",      ["a"], [("a", 0.5)]),
        ("q2",         [], [("a", 0.5)]),
        ("q3",      ["a"], [("b", 0.5), ("a", 0.4)]),
    ]
    # fmt: on

    # convert shorthand to response wrapper
    responses = [
        Response(
            id=qid,
            question="",
            reference_answer="",
            sources=qrels,
            generated_answer="",
            latency=0,
            contexts=[
                Context(content="", score=score, url=url) for (url, score) in run
            ],
        )
        for (qid, qrels, run) in responses
    ]

    mrr = MeanReciprocalRank()
    assert mrr.compute(responses) == {"score": 0.5, "raw": [1.0, 0.0, 0.5]}


def test_recall_at_cutoff():
    # fmt: off
    # (query_id, sources, list of retrieved contexts: (url, score))
    responses = [
        ("q1",      ["a"], [("a", 0.5), ("b", 0.4), ("c", 0.3)]),
        ("q2",         [], [("a", 0.5), ("b", 0.4), ("c", 0.3)]),
        ("q3", ["b", "c"], [("a", 0.5), ("c", 0.4), ("b", 0.3)]),
        ("q4", ["b", "c"], []),
    ]
    # fmt: on

    # convert shorthand to response wrapper
    responses = [
        Response(
            id=qid,
            question="",
            reference_answer="",
            sources=qrels,
            generated_answer="",
            latency=0,
            contexts=[
                Context(content="", score=score, url=url) for (url, score) in run
            ],
        )
        for (qid, qrels, run) in responses
    ]

    r_at_1 = RecallAtCutoff(cutoff=1)
    r_at_3 = RecallAtCutoff(cutoff=3)
    assert r_at_1.compute(responses) == {"score": 1 / 3, "raw": [1, np.nan, 0, 0]}
    assert r_at_3.compute(responses) == {"score": 2 / 3, "raw": [1, np.nan, 1, 0]}


def test_precision_at_cutoff():
    # fmt: off
    # (query_id, sources, list of retrieved contexts: (url, score))
    responses = [
        ("q1",      ["a"], [("a", 0.5), ("b", 0.4), ("c", 0.3)]),   # 1 relevant in top 1, 1 relevant in top 3
        ("q2",         [], [("a", 0.5), ("b", 0.4), ("c", 0.3)]),   # no relevant docs, precision=0
        ("q3", ["b", "c"], [("a", 0.5), ("c", 0.4), ("b", 0.3)]),   # 0 relevant in top 1, 2 relevant in top 3
        ("q4", ["b", "c"], []),                                     # no retrieved docs, precision=0
    ]
    # fmt: on

    # Convert shorthand to Response wrapper
    responses = [
        Response(
            id=qid,
            question="",
            reference_answer="",
            sources=qrels,
            generated_answer="",
            latency=0,
            contexts=[
                Context(content="", score=score, url=url) for (url, score) in run
            ],
        )
        for (qid, qrels, run) in responses
    ]

    p_at_1 = PrecisionAtCutoff(cutoff=1)
    p_at_3 = PrecisionAtCutoff(cutoff=3)

    assert p_at_1.compute(responses) == {"score": 1 / 4, "raw": [1, 0, 0, 0]}
    assert p_at_3.compute(responses) == {"score": 1 / 4, "raw": [1 / 3, 0, 2 / 3, 0]}


def test_non_answer_critic():
    metric = NonAnswerCriticClassic()
    assert metric.compute_one("The answer is 42")
    assert not metric.compute_one("Unfortunately, I do not have any knowledge")
    assert not metric.compute_one("Unfortunately, the documents do not specify this.")

    responses = [
        {"generated_answer": ["The answer is 42"]},
        {"generated_answer": ["Unfortunately, I do not have any knowledge"]},
        {"generated_answer": ["Unfortunately, the documents do not specify this."]},
    ]

    result = metric.compute(responses)
    assert result["raw"] == [True, False, False]
    assert result["score"] == 1 / 3
