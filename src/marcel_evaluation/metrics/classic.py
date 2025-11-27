import re
from typing import Any, Dict, List

import bert_score
import ir_measures
import numpy as np
import sacrebleu
from nltk import word_tokenize
from rouge_score import rouge_scorer

from marcel_evaluation.base import Metric, Response


class GeneratedAnswerLength(Metric):
    @property
    def name(self):
        return "GeneratedAnswerLength"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        lengths = [len(word_tokenize(r["generated_answer"][0])) for r in responses]
        score = np.mean(lengths)
        return {"score": score, "raw": lengths}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class ReferenceAnswerLength(Metric):
    @property
    def name(self):
        return "ReferenceAnswerLength"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def compute(self, responses: List[Response]):
        lengths = [len(word_tokenize(r["reference_answer"])) for r in responses]
        score = np.mean(lengths)
        return {"score": score, "raw": lengths}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class ContextLength(Metric):
    @property
    def name(self):
        return "ContextLength"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def compute(self, responses: List[Response]):
        lengths = []

        for response in responses:
            context_length = 0
            for context in response["contexts"]:
                context_length += len(word_tokenize(context["content"]))
            lengths.append(context_length)

        score = np.mean(lengths)
        return {"score": score, "raw": lengths}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class CorpusBLEU(Metric):
    @property
    def name(self):
        return "CorpusBLEU"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        bleu_result = sacrebleu.corpus_bleu(
            hypotheses=[r["generated_answer"][0] for r in responses],
            references=[[r["reference_answer"] for r in responses]],
        )
        return {
            # divide by 100 to match with other metrics
            "score": bleu_result.score / 100
        }

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class ROUGE(Metric):
    def __init__(
        self, rouge_type, use_stemmer=False, split_summaries=False, tokenizer=None
    ):
        assert rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=[rouge_type],
            use_stemmer=use_stemmer,
            split_summaries=split_summaries,
            tokenizer=tokenizer,
        )
        self.rouge_type = rouge_type

    @property
    def name(self):
        return self.rouge_type

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        scores = []
        for response in responses:
            score = self.scorer.score(
                target=response["reference_answer"],
                prediction=response["generated_answer"][0],
            )
            scores.append(score[self.rouge_type].fmeasure)

        agg = np.mean(scores)
        return {"score": agg, "raw": scores}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class BERTScore(Metric):
    def __init__(self, **metric_args):
        self.scorer = bert_score.BERTScorer(**metric_args)

    @property
    def name(self):
        return "BERTScore"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        P, R, F1 = self.scorer.score(
            cands=[r["generated_answer"][0] for r in responses],
            refs=[r["reference_answer"] for r in responses],
        )
        return {
            "score": np.mean(F1.tolist()),  # type: ignore
            "raw": F1.tolist(),  # type: ignore
            "precision": P.tolist(),  # type: ignore
            "recall": R.tolist(),  # type: ignore
        }

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


def run_to_trec(responses: List[Response]):
    """
    Converts responses into TREC format.

    Example
    -------
    qrels = {
        "Q0": {"D0": 0, "D1": 1},
        "Q1": {"D0": 0, "D3": 2}
    }
    run = {
        "Q0": {"D0": 1.2, "D1": 1.0},
        "Q1": {"D0": 2.4, "D3": 3.6}
    }
    """

    qrels, run = {}, {}

    for response in responses:
        query_id = response["id"]
        if len(response["sources"]) == 0:
            # Fixes broken score calculation when one or more qrel sets are empty
            # See: https://github.com/cvangysel/pytrec_eval/issues/49
            relevant = {"DUMMY": 1}
        else:
            relevant = {source: 1 for source in response["sources"]}
        retrieved = {
            context["url"]: context["score"] for context in response["contexts"]
        }
        qrels[query_id] = relevant
        run[query_id] = retrieved

    return qrels, run


class MeanReciprocalRank(Metric):
    @property
    def name(self):
        return "MeanReciprocalRank"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def compute(self, responses: List[Response]):
        qrels, run = run_to_trec(responses)
        mrr = ir_measures.MRR()
        scores = [score.value for score in mrr.iter_calc(qrels, run)]
        return {"score": np.mean(scores), "raw": scores}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class PrecisionAtCutoff(Metric):
    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    @property
    def name(self):
        return f"p@{self.cutoff}"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def compute(self, responses: List[Response]):
        qrels, run = run_to_trec(responses)
        metric = ir_measures.P(cutoff=self.cutoff)
        scores = [score.value for score in metric.iter_calc(qrels, run)]
        return {"score": np.mean(scores), "raw": scores}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class RecallAtCutoff(Metric):
    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    @property
    def name(self):
        return f"r@{self.cutoff}"

    @property
    def uses_generated_answer(self) -> bool:
        return False

    def compute(self, responses: List[Response]):
        qrels, run = run_to_trec(responses)
        metric = ir_measures.R(cutoff=self.cutoff)
        scores = [score.value for score in metric.iter_calc(qrels, run)]

        # set score to nan if query does not have any relevant documents
        scores = [
            score if len(response["sources"]) > 0 else np.nan
            for score, response in zip(scores, responses)
        ]

        return {"score": np.nanmean(scores), "raw": scores}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError


class NonAnswerCriticClassic(Metric):
    def __init__(self):
        self._non_answer_pattern = re.compile(
            r"(I (do not|don't) have (any )?(knowledge|information)|"
            r"The text doesn't provide information|"
            r"I do not have this information|"
            r"the documents do not specify)",
            re.IGNORECASE,
        )

    @property
    def name(self):
        return "NonAnswerCriticClassic"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute_one(self, text: str) -> bool:
        match = self._non_answer_pattern.search(text)
        if match and len(text.split(" ")) < 30:
            return False
        return True

    def compute(self, responses: List[Response]) -> Dict[str, Any]:
        scores = [self.compute_one(r["generated_answer"][0]) for r in responses]
        agg = np.mean(scores)
        return {"score": agg, "raw": scores}

    async def compute_async(self, responses: List[Response]):
        raise NotImplementedError
