from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from marcel_evaluation.metrics.answer_similarity import AnswerSimilarity


@pytest.fixture
def mock_model():
    def mock_encode(sentences: str | list[str], **kwargs) -> np.ndarray:
        """
        We simply one-hot encode the sentences; if a sentence contains a keyword, the corresponding one-hot
        encoding is added to the sentence embedding.

        Adapted from: https://github.com/huggingface/sentence-transformers/blob/85ec64559f4414aa536eca4bf53538291e0a333f/tests/evaluation/test_information_retrieval_evaluator.py#L16
        """
        one_hot_encodings = {
            "pokemon": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "car": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            "vehicle": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            "fruit": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            "vegetable": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
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


def test_compute(mock_model):
    metric = AnswerSimilarity(model=mock_model)
    responses = [
        {"generated_answer": ["pokemon"], "reference_answer": "car"},
        {"generated_answer": ["car"], "reference_answer": "car"},
        {"generated_answer": ["fruit"], "reference_answer": "fruit"},
        {"generated_answer": [""], "reference_answer": "fruit"},
    ]
    result = metric.compute(responses)
    assert result["raw"] == [0, 1, 1, 0]
    assert result["score"] == 2 / 4
