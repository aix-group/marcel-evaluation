from typing import List, Literal, Optional

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel

from marcel_evaluation.llm import batch_generate


class Verdict(BaseModel):
    statement: str
    reason: str
    verdict: Literal[0, 1]


class Output(BaseModel):
    verdicts: List[Verdict]


class Example(BaseModel):
    context: str
    statements: List[str]
    output: Optional[Output] = None


INSTRUCTION = """
## Instruction
Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

{{ output_model.model_json_schema() }}

Only respond with the JSON, nothing else.

## Examples
{% for example in examples %}
Context: {{ example.context }}
Statements: {{ example.statements }}
Answer: {{ example.model_dump()['output'] }}

{% endfor %}

## Task
Context: {{ task.context }}
Statements: {{ task.statements }}
Answer:
""".strip()

EXAMPLE_1 = Example(
    context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
    statements=[
        "John is majoring in Biology.",
        "John is taking a course on Artificial Intelligence.",
        "John is a dedicated student.",
        "John has a part-time job.",
    ],
    output=Output(
        verdicts=[
            Verdict(
                statement="John is majoring in Biology.",
                reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                verdict=0,
            ),
            Verdict(
                statement="John is taking a course on Artificial Intelligence.",
                reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                verdict=0,
            ),
            Verdict(
                statement="John is a dedicated student.",
                reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                verdict=1,
            ),
            Verdict(
                statement="John has a part-time job.",
                reason="There is no information given in the context about John having a part-time job.",
                verdict=0,
            ),
        ]
    ),
)

EXAMPLE_2 = Example(
    context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
    statements=["Albert Einstein was a genius."],
    output=Output(
        verdicts=[
            Verdict(
                statement="Albert Einstein was a genius.",
                reason="The context and statement are unrelated",
                verdict=0,
            ),
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


class ClaimEntailmentVerifier:
    def __init__(self, model: str, temperature: float = 0, max_tokens: int = 8192):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompts(
        self, texts: list[str], claims_per_text: list[list[str]]
    ) -> list[list[dict[str, str]]]:
        prompts = []
        for text, claims in zip(texts, claims_per_text):
            task = Example(context=text, statements=claims)
            prompt = format_prompt(task)
            prompts.append(prompt)
        return prompts

    async def calculate_entailment(
        self, texts: list[str], claims_per_text: list[list[str]]
    ) -> list[list[int]]:
        prompts = self.build_prompts(texts, claims_per_text)
        outputs = await batch_generate(
            prompts,
            model=self.model,
            response_format=Output,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_concurrency=5,
        )

        return [[verdict.verdict for verdict in output.verdicts] for output in outputs]


if __name__ == "__main__":
    task = Example(
        context="As of July 2024, the President of the United States is Joseph R. Biden Jr. He assumed office as the 46th President on January 20, 2021, after defeating the incumbent, Donald Trump, in the 2020 presidential election.",
        statements=[
            "Joseph Biden is the 46th president of the US.",
            "Joseph Biden won the election.",
            "The NATO conference 2024 will be hosted in the US.",
        ],
    )
    prompt = format_prompt(task)
    print(prompt[0]["content"])
