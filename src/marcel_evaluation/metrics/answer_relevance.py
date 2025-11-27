from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.llm import batch_generate
from marcel_evaluation.metrics import answer_relevance_qgen
from marcel_evaluation.metrics.non_answer_critic import NonAnswerCritic


class QuestionGenerator:
    def __init__(self, model, n=5, temperature=0.7, max_tokens=2048):
        assert n > 1, "n must be greater than 1"
        self.model = model
        self.n = n
        self.temperature = temperature
        self.max_tokens = max_tokens

    def format_contexts(self, contexts):
        # concatenates all contexts and escapes special characters including newlines
        text = "\n".join([context["content"] for context in contexts])
        return repr(text)

    async def generate_questions(self, responses: List[Response]):
        prompts = [
            answer_relevance_qgen.format_prompt(
                answer_relevance_qgen.Example(
                    context=self.format_contexts(response["contexts"]),
                    answer=response["generated_answer"][0],
                )
            )
            for response in responses
        ]

        result = await batch_generate(
            conversations=prompts,
            model=self.model,
            response_format=answer_relevance_qgen.GeneratedQuestion,
            n=self.n,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )

        return [[question.question for question in generation] for generation in result]


class AnswerRelevance(Metric):
    """How relevant the answer is to the question."""

    def __init__(
        self,
        model: str,
        non_answer_critic=None,
        question_generator=None,
        embedding_model_or_name: Union[SentenceTransformer, str] = "all-mpnet-base-v2",
    ):
        if isinstance(embedding_model_or_name, str):
            self.embedding_model = SentenceTransformer(embedding_model_or_name)
        else:
            self.embedding_model = embedding_model_or_name

        if non_answer_critic:
            self.non_answer_critic = non_answer_critic
        else:
            self.non_answer_critic = NonAnswerCritic(model=model)

        if question_generator:
            self.question_generator = question_generator
        else:
            self.question_generator = QuestionGenerator(
                n=5, temperature=0.7, model=model
            )

    @property
    def name(self):
        return "AnswerRelevance"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def question_similarity(self, question: str, generated: List[str]):
        q_embedding = self.embedding_model.encode(question)
        gen_embeddings = self.embedding_model.encode(generated)
        similarities = self.embedding_model.similarity(q_embedding, gen_embeddings)
        return similarities.mean().item()

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        # Step 1: identify responses with non-answers
        # Output of non-answer critic: 0 = answer, 1 = non-answer
        result = await self.non_answer_critic.compute_async(responses)
        noncomittal = result["raw"]
        committal_responses = [
            response for response, label in zip(responses, noncomittal) if label == 0
        ]
        committal_ixs = [i for i, label in enumerate(noncomittal) if label == 0]

        if len(committal_responses) == 0:
            return {
                "score": 0.0,
                "raw": None,
                "noncommittal": noncomittal,
                "generated_questions": [],
            }

        # Step 2: generate questions (only for valid answers)
        generated_questions = await self.question_generator.generate_questions(
            committal_responses
        )

        # Step 3: Relevance calculation
        # Make sure that we return raw scores and generated_questions for all responses.
        # Non-answers will get a None score/generated_question
        scores = [np.nan] * len(responses)
        all_questions = [[] for _ in range(len(responses))]
        for ix, response, questions in zip(
            committal_ixs, committal_responses, generated_questions
        ):
            score = self.question_similarity(response["question"], questions)
            scores[ix] = score
            all_questions[ix] = questions

        return {
            "score": np.nanmean(scores),
            "raw": scores,
            "noncommittal": noncomittal,
            "generated_questions": all_questions,
        }
