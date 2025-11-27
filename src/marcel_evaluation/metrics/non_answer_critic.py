from typing import List

from marcel_evaluation.base import Metric, Response
from marcel_evaluation.llm import batch_generate
from marcel_evaluation.metrics import non_answer_critic_prompt


class NonAnswerCritic(Metric):
    """Check if generated_answer is a non-answer."""

    def __init__(self, model: str):
        self.model = model
        self.temperature = 0
        self.max_tokens = 2048

    @property
    def name(self):
        return "NonAnswerCritic"

    @property
    def uses_generated_answer(self) -> bool:
        return True

    def compute(self, responses: List[Response]):
        raise NotImplementedError

    async def compute_async(self, responses: List[Response]):
        # Step 1: Prompts preparation
        prompts = [
            non_answer_critic_prompt.format_prompt(
                non_answer_critic_prompt.Example(
                    answer=response["generated_answer"][0],
                )
            )
            for response in responses
        ]

        # Step 2: Context verification
        answer_results = await batch_generate(
            conversations=prompts,
            model=self.model,
            response_format=non_answer_critic_prompt.ClassifiedAnswer,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )

        # Step 3: Computation
        good_answers = [not answer.non_answer for answer in answer_results]

        return {
            "score": sum(good_answers) / len(answer_results),
            "raw": [answer.non_answer for answer in answer_results],
            "rationales": [answer.rationale for answer in answer_results],
        }
