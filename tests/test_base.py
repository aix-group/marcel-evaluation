from marcel_evaluation.base import MultiGenerationMetricWrapper
from marcel_evaluation.metrics.classic import GeneratedAnswerLength


def test_normalize_runs():
    wrapper = MultiGenerationMetricWrapper(metric=None)

    run = [
        {"question": "a", "generated_answer": ["a_1", "a_2"]},
        {"question": "b", "generated_answer": ["b_1", "b_2"]},
        {"question": "c", "generated_answer": ["c_1", "c_2"]},
    ]

    runs = wrapper._normalize_runs(run)
    assert len(runs) == 2
    assert runs[0] == [
        {"question": "a", "generated_answer": ["a_1"]},
        {"question": "b", "generated_answer": ["b_1"]},
        {"question": "c", "generated_answer": ["c_1"]},
    ]
    assert runs[1] == [
        {"question": "a", "generated_answer": ["a_2"]},
        {"question": "b", "generated_answer": ["b_2"]},
        {"question": "c", "generated_answer": ["c_2"]},
    ]


def test_aggregate_scores():
    wrapper = MultiGenerationMetricWrapper(metric=None)

    run_results = [
        {"score": 3.0, "raw": [5, 2, 2]},  # first run
        {"score": 4.0, "raw": [6, 3, 3]},  # second run
    ]

    assert wrapper._aggregate_scores(run_results) == {
        "score": 3.5,
        "raw": [
            {"score": 5.5, "raw": [5, 6]},
            {"score": 2.5, "raw": [2, 3]},
            {"score": 2.5, "raw": [2, 3]},
        ],
    }


def test_multi_generation_metric_wrapper():
    responses = [
        {"generated_answer": ["a b c d e", "a b c d e f"]},
        {"generated_answer": ["a b", "a b c"]},
        {"generated_answer": ["a b", "a b c"]},
    ]

    metric = GeneratedAnswerLength()
    metric = MultiGenerationMetricWrapper(metric)
    assert metric.compute(responses) == {
        "score": 3.5,
        "raw": [
            {"score": 5.5, "raw": [5, 6]},
            {"score": 2.5, "raw": [2, 3]},
            {"score": 2.5, "raw": [2, 3]},
        ],
    }
