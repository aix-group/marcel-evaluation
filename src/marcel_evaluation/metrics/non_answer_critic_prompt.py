from typing import Literal, Optional

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


class ClassifiedAnswer(BaseModel):
    rationale: str
    non_answer: Literal[0, 1]


class Example(BaseModel):
    answer: str
    classification: Optional[ClassifiedAnswer] = None


INSTRUCTION = """
## Instruction
Classify the given response as either an answer or a non-answer.

Non-answers are responses that indicate a lack of information or inability to provide an answer. They typically include phrases such as:
- "I don't know..."
- "I can't provide information about..."
- "It is not stated..."
- "There is no answer"
- "There is no information..."
- Or other synonymous expressions indicating uncertainty or lack of information

Any response that provides concrete information or a direct answer, even if brief, should be classified as an answer.

For each response, provide a classification in the following format:
Classification: {'rationale': 'Brief explanation of why it's an answer or non-answer', 'non_answer': 0 or 1}

Use 0 for answers and 1 for non-answers in the 'non_answer' field.

## Examples
{% for example in examples %}
Response: {{ example.answer }}
Classification: {{ example.model_dump()['classification'] }}

{% endfor %}

Now, classify the following response:
Response: {{ task.answer }}
Classification:
""".strip()

EXAMPLE_1 = Example(
    answer="The information provided does not mention anything about compulsory elective modules or the number of free modules a student can choose from in the M.Sc. Data Science program at Marburg University.",
    classification=ClassifiedAnswer(
        rationale="The response indicates a lack of information, which is characteristic of a non-answer.",
        non_answer=1,
    ),
)

EXAMPLE_2 = Example(
    answer="The question is not clear because there is no specific module mentioned. The text only talks about enrollment as a doctoral student and provides information about the required documents and the enrollment process. It does not mention a specific module.",
    classification=ClassifiedAnswer(
        rationale="The response states that specific information is not provided, making it a non-answer.",
        non_answer=1,
    ),
)


EXAMPLE_3 = Example(
    answer="Master of Science (M.Sc.)",
    classification=ClassifiedAnswer(
        rationale="The response provides a specific piece of information, making it a valid answer.",
        non_answer=0,
    ),
)

JINJA_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)
JINJA_TEMPLATE = JINJA_ENV.from_string(
    INSTRUCTION, globals={"examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3]}
)


def format_prompt(task: Example):
    content = JINJA_TEMPLATE.render(task=task)
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    task = Example(
        answer="The degree for this Data Science program is a Master's degree (M.Sc.)."
    )
    prompt = format_prompt(task)
    print(prompt[0]["content"])
