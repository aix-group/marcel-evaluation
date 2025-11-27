from collections import defaultdict
from typing import List, Optional

import nltk
from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel

from marcel_evaluation.base import Response
from marcel_evaluation.llm import batch_generate


class Output(BaseModel):
    simpler_statements: List[str]


class Example(BaseModel):
    question: str
    answer: str
    sentence: str
    output: Optional[Output] = None


INSTRUCTION = """
## Instruction
You are given a question, an answer, and one sentence extracted from the answer. Please break down the sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

{{ output_model.model_json_schema() }}

## Examples
{% for example in examples %}
Question: {{ example.question }}
Answer: {{ example.answer }}
Sentence: {{ example.sentence }}
Analysis: {{ example.model_dump()['output'] }}

{% endfor %}

## Task
Question: {{ task.question }}
Answer: {{ task.answer }}
Sentence: {{ task.sentence }}
Analysis:
""".strip()

EXAMPLE_1 = Example(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentence="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",
    output=Output(
        simpler_statements=[
            "Albert Einstein was a German-born theoretical physicist.",
            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
        ]
    ),
)

EXAMPLE_2 = Example(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    sentence="He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
    output=Output(
        simpler_statements=[
            "Albert Einstein was best known for developing the theory of relativity.",
            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
        ]
    ),
)


JINJA_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)
JINJA_TEMPLATE = JINJA_ENV.from_string(
    INSTRUCTION,
    globals={
        "examples": [EXAMPLE_1, EXAMPLE_2],
        "output_model": Output,
    },
)


def format_prompt(task: Example):
    content = JINJA_TEMPLATE.render(task=task)
    return [{"role": "user", "content": content}]


class ClaimSplitter:
    """
    Splits answers into a list of atomic claims.
    """

    def __init__(self, model: str, temperature: float = 0, max_tokens: int = 8192):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def split_sentences(self, text: str) -> list[str]:
        return nltk.sent_tokenize(text)

    def build_prompts(self, responses: list[Response]) -> tuple[list[str], list[int]]:
        prompts, response_indices = [], []
        for i, response in enumerate(responses):
            question = response["question"]
            answer = response["generated_answer"][0]
            sentences = self.split_sentences(answer)

            for sentence in sentences:
                task = Example(question=question, answer=answer, sentence=sentence)
                prompt = format_prompt(task)
                prompts.append(prompt)
                response_indices.append(i)

        return prompts, response_indices

    def group_claims_by_response(
        self,
        claims_per_sentence: list[list[str]],
        response_indices: list[int],
        n_responses: int,
    ) -> list[list[str]]:
        claims_per_response = defaultdict(list)
        for claims, i in zip(claims_per_sentence, response_indices):
            claims_per_response[i].extend(claims)
        return [claims_per_response[i] for i in range(n_responses)]

    async def extract_claims(self, responses: list[Response]) -> list[list[str]]:
        prompts, response_indices = self.build_prompts(responses)
        outputs = await batch_generate(
            prompts,
            model=self.model,
            response_format=Output,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )
        return self.group_claims_by_response(
            claims_per_sentence=[output.simpler_statements for output in outputs],
            response_indices=response_indices,
            n_responses=len(responses),
        )


if __name__ == "__main__":
    task = Example(
        question="Who is the president of the united states?",
        answer="As of July 2024, the President of the United States is Joseph R. Biden Jr. He assumed office as the 46th President on January 20, 2021, after defeating the incumbent, Donald Trump, in the 2020 presidential election.",
        sentence="As of July 2024, the President of the United States is Joseph R. Biden Jr.",
    )
    prompt = format_prompt(task)
    print(prompt[0]["content"])
