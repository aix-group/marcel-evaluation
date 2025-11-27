from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.metrics import answer_relevance


class MockNonAnswerCritic(Metric):
    @property
    def name(self):
        return "MockNonAnswerCritic"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        def is_non_answer(response):
            return "i dont know" in response["generated_answer"][0].lower()

        labels = [is_non_answer(response) for response in responses]
        total = len(responses)
        abstain = sum(labels)
        score = total - abstain / total

        return {"score": score, "raw": labels}


class MockQuestionGenerator(answer_relevance.QuestionGenerator):
    async def generate_questions(self, responses: List[Response]):
        return [[response["question"] for _ in range(self.n)] for response in responses]


@pytest.fixture
def mock_model():
    def mock_encode(sentences: str | list[str], **kwargs) -> np.ndarray:
        """
        We simply one-hot encode the sentences; if a sentence contains a keyword, the corresponding one-hot
        encoding is added to the sentence embedding.

        Adapted from: https://github.com/huggingface/sentence-transformers/blob/85ec64559f4414aa536eca4bf53538291e0a333f/tests/evaluation/test_information_retrieval_evaluator.py#L16
        """
        one_hot_encodings = {
            "capital": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "mountain": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        }
        if isinstance(sentences, str):
            sentences = [sentences]
        embeddings = []
        for sentence in sentences:
            encoding = torch.zeros(5)
            for keyword, one_hot in one_hot_encodings.items():
                if keyword in sentence:
                    encoding += one_hot
            embeddings.append(encoding)
        return np.stack(embeddings)

    model = Mock(spec=SentenceTransformer)
    model.similarity_fn_name = "cosine"
    model.similarity.side_effect = cos_sim
    model.encode.side_effect = mock_encode
    return model


@pytest.mark.asyncio
async def test_compute(mock_model):
    responses = [
        {
            "question": "What is the capital of Germany?",
            "generated_answer": ["Berlin"],
            "contexts": [
                {"content": "The capital of Germany is Berlin."},
            ],
        },
        {
            "question": "What is the tallest mountain of Norway?",
            "generated_answer": ["Galdhøpiggen"],
            "contexts": [
                {
                    "content": "The tallest mountain in Norway is Galdhøpiggen with 2,469 m of elevation."
                }
            ],
        },
        {
            "question": "What is the answer to everything?",
            "generated_answer": ["i dont know"],
            "contexts": [{"content": "context"}],
        },
    ]

    metric = answer_relevance.AnswerRelevance(
        model=None,
        embedding_model_or_name=mock_model,
        non_answer_critic=MockNonAnswerCritic(),
        question_generator=MockQuestionGenerator(model=None, n=5),
    )

    result = await metric.compute_async(responses)
    assert result["score"] == 1
    assert result["raw"] == [1, 1, np.nan]
    assert result["noncommittal"] == [0, 0, 1]
    assert len(result["generated_questions"]) == 3
    assert len(result["generated_questions"][0]) == 5
    assert len(result["generated_questions"][1]) == 5
    assert len(result["generated_questions"][2]) == 0


@pytest.mark.asyncio
async def test_compute_all_noncomittal():
    responses = [
        {
            "question": "What is the answer to everything?",
            "generated_answer": ["i dont know"],
            "contexts": [{"content": "context"}],
        }
    ]
    metric = answer_relevance.AnswerRelevance(
        model=None,
        embedding_model_or_name=None,
        non_answer_critic=MockNonAnswerCritic(),
        question_generator=None,
    )
    result = await metric.compute_async(responses)
    assert result["score"] == 0
    assert result["raw"] is None
    assert result["noncommittal"] == [1]
