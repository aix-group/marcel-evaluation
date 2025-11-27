from typing import List, Literal, Optional

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


class ClassifiedSentence(BaseModel):
    sentence: str
    reason: str
    label: Literal[0, 1]


class ClassifiedSentencesList(BaseModel):
    sentences: List[ClassifiedSentence]


class Example(BaseModel):
    question: str
    context: str
    answer: str
    classification: Optional[ClassifiedSentencesList] = None


INSTRUCTION = """
## Instruction
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Provide the answer sentence, a rationale for your decision and the final classification label. Use only 1 (Yes) or 0 (No) as a binary classification.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

{{ output_model.model_json_schema() }}

## Examples
{% for example in examples %}
Question: "{{ example.question }}"
Context: "{{ example.context }}"
Answer: "{{ example.answer }}"
Classification: {{ example.model_dump()['classification'] }}

{% endfor %}

## Task
Question: "{{ task.question }}"
Context: "{{ task.context }}"
Answer: "{{ task.answer }}"
Classification:
""".strip()


EXAMPLE_1 = Example(
    question="What can you tell me about albert Albert Einstein?",
    context="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
    answer="Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
                reason="The date of birth of Einstein is mentioned clearly in the context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",
                reason="The exact sentence is present in the given context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He published 4 papers in 1905.",
                reason="There is no mention about papers he wrote in the given context.",
                label=0,
            ),
            ClassifiedSentence(
                sentence="Einstein moved to Switzerland in 1895.",
                reason="There is no supporting evidence for this in the given context.",
                label=0,
            ),
        ]
    ),
)


EXAMPLE_2 = Example(
    question="who won 2020 icc world cup?",
    context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
    answer="England",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="England",
                reason="From context it is clear that England defeated Pakistan to win the World Cup.",
                label=1,
            )
        ]
    ),
)

EXAMPLE_3 = Example(
    question="What is the primary fuel for the Sun?",
    context="",
    answer="Hydrogen",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Hydrogen",
                reason="The context contains no information",
                label=0,
            )
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
        "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
        "output_model": ClassifiedSentencesList,
    },
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
