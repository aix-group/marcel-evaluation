from typing import Optional

from jinja2 import Environment, StrictUndefined
from pydantic import BaseModel


class GeneratedQuestion(BaseModel):
    question: str


class Example(BaseModel):
    context: str
    answer: str
    output: Optional[GeneratedQuestion] = None


INSTRUCTION = """
## Instruction
Consider the given answer and context. Please generate a question for the answer. The question should be specific, such that the answer is most likely the real answer.

The output should be a well-formatted JSON instance that conforms to the JSON schema below.

{{ output_model.model_json_schema() }}

## Examples
{% for example in examples %}
Context: "{{ example.context }}"
Answer: "{{ example.answer }}"
Output: {{ example.model_dump()['output'] }}

{% endfor %}

## Task
Context: "{{ task.context }}"
Answer: "{{ task.answer }}"
Output:
""".strip()

EXAMPLE_1 = Example(
    answer="Albert Einstein was born in Germany.",
    context="Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time",
    output=GeneratedQuestion(question="Where was Albert Einstein born?"),
)

EXAMPLE_2 = Example(
    answer="It can change its skin color based on the temperature of its environment.",
    context="A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.",
    output=GeneratedQuestion(
        question="What unique ability does the newly discovered species of frog have?",
    ),
)

EXAMPLE_3 = Example(
    answer="The tallest mountain on earth is Mt. Everest.",
    context="The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.",
    output=GeneratedQuestion(question="What is the tallest mountain on Earth?"),
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
        "output_model": GeneratedQuestion,
    },
)


def format_prompt(task: Example):
    content = JINJA_TEMPLATE.render(task=task)
    return [{"role": "user", "content": content}]


if __name__ == "__main__":
    task = Example(
        context=repr(
            "Main Content\n\n# Perspectives\n\nAre you interested in the M.Sc. Data Science program and wondering what you can do with the degree? Or you are already close to graduation and wonder what now? Below you will find information about career prospects.\n\nAlle Elemente ausklappen Alle Elemente einklappen \n\n  * ### [Inhalt ausklappen Inhalt einklappen Qualifications][654]\n\nThe master's degree program in Data Science prepares you for future work in the field of data science in science and industry, for which in-depth knowledge of computer science as well as applied mathematics is required. The study program serves to deepen and specialize the previous bachelor's degree. In addition to the technical skills, such as designing efficient algorithms and implementing them in distributed environments, you will acquire many other qualifications during your studies that you can use in your later professional life.\n\n**Project Work**\n\nThrough the project-oriented structure of the program, you will learn to work on difficult issues both independently and scientifically as part of a team. You will be able to plan larger projects over a longer period of time and complete them on time. You will then be able to present your results in a professional manner and adapted to your target group.\n\n**Key Qualifications**\n\nIn addition, you will learn how to assess yourself: How well can you work under time pressure and organize yourself? What are your strengths, what are your weaknesses? These insights will help you and your future employers or business partners. As a student of the Data Science program, you will be guided to teamwork and independent learning in the various internships and exercises, which will increasingly promote social and communication skills.\n\n  * ### [Inhalt ausklappen Inhalt einklappen Fields of Activity][654]\n\n**Exzellent Career Prospects**\n\nIn almost all scientific disciplines as well as in industry and business, the collection of massive amounts of data results in problems when dealing with this data. Many disciplines and companies lack the knowledge and tools to deal with these data volumes and to derive their benefits from them. Well-educated graduates of the Data Science program can perform important functions here because they have been specifically trained to solve such problems - there are therefore excellent career prospects!\n\n**Big Data - everywhere!**\n\nBig data is now present in almost all areas of life, even if this is often not perceived. The fields of activity for data analysts are therefore diverse and interdisciplinary - they can be found in research institutions as well as in virtually all areas of industry and business, such as in the banking and insurance sectors, the automotive industry or innovative companies in the environment of the internet.\n\n  * ### [Inhalt ausklappen Inhalt einklappen Doctorate][654]\n\nIn your master's degree program, you have learned to work scientifically on your own. In addition, you have acquired in-depth expertise in a special field of computer science or applied mathematics. If you are interested in working on long-term issues (such as your master's thesis), you can further deepen your knowledge and skills with us through subsequent doctoral studies and work on a current research problem.\n\nIf you are interested in a doctorate, please contact the professor whose field interests you the most."
        ),
        answer="The degree for this Data Science program is a Master's degree (M.Sc.).",
    )
    prompt = format_prompt(task)
    print(prompt[0]["content"])
