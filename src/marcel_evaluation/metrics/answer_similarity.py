import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from marcel_evaluation.base import Metric, Response

logger = logging.getLogger(__name__)


class AnswerSimilarity(Metric):
    """Estimate the similarity between answer and reference embeddings."""

    def __init__(self, model: Union[str, SentenceTransformer]):
        if isinstance(model, SentenceTransformer):
            self.model = model
        else:
            self.model = SentenceTransformer(model)

    @property
    def name(self):
        return "AnswerSimilarity"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        gen = [r["generated_answer"][0] for r in responses]
        ref = [r["reference_answer"] for r in responses]
        gen_emb = self.model.encode(gen)
        ref_emb = self.model.encode(ref)
        similarity_matrix = self.model.similarity(gen_emb, ref_emb)
        similarities = similarity_matrix.diagonal().tolist()
        return {"score": np.mean(similarities), "raw": similarities}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError
