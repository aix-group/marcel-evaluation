from typing import Literal, Optional

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


class Verdict(BaseModel):
    reason: str
    verdict: Literal[0, 1]


class Example(BaseModel):
    question: str
    context: str
    answer: str
    verification: Optional[Verdict] = None


INSTRUCTION = """
## Instruction
Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

{{ output_model.model_json_schema() }}

## Examples
{% for example in examples %}
Question: {{ example.question }}
Context: {{ example.context }}
Answer: {{example.answer}}
Verification: {{ example.model_dump()['verification'] }}

{% endfor %}

## Task
Question: {{ task.question }}
Context: {{ task.context }}
Answer: {{task.answer}}
Verification:
""".strip()


EXAMPLE_1 = Example(
    question="What is the tallest mountain in the world?",
    context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
    answer="Mount Everest.",
    verification=Verdict(
        reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
        verdict=0,
    ),
)
EXAMPLE_2 = Example(
    question="who won 2020 icc world cup?",
    context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
    answer="England",
    verification=Verdict(
        reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
        verdict=1,
    ),
)

EXAMPLE_3 = Example(
    question="What can you tell me about albert Albert Einstein?",
    context="""Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
    answer="Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895",
    verification=Verdict(
        reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
        verdict=1,
    ),
)

JINJA_ENV = Environment(
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)
JINJA_TEMPLATE = JINJA_ENV.from_string(
    INSTRUCTION,
    globals={"examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3], "output_model": Verdict},
)


def format_prompt(task: Example):
    content = JINJA_TEMPLATE.render(task=task)
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    task = Example(
        question="What is the degree for this Data Science program?",
        context="Are you interested in the M.Sc. Data Science program and wondering what you can do with the degree?",
        answer="The degree for this Data Science program is a Master's degree (M.Sc.).",
    )
    prompt = format_prompt(task)
    print(prompt[0]["content"])
