import os

import pytest
from pydantic import BaseModel

from marcel_evaluation import llm


@pytest.mark.skipif(
    not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"),
    reason="Export an env var called OPENAI_BASE_URL and OPENAI_API_KEY to run this test.",
)
@pytest.mark.asyncio
async def test_batch_generate():
    class Answer(BaseModel):
        country: str
        capital: str

    conversations = [
        [
            {
                "role": "user",
                "content": 'What is the captial of Germany? Respond in json: {"country": "", "capital": ""}',
            },
        ],
        [
            {
                "role": "user",
                "content": 'What is the captial of France? Respond in json: {"country": "", "capital": ""}',
            }
        ],
    ]

    response = await llm.batch_generate(
        conversations,
        "openai/gpt-oss-20b",
        response_format=Answer,
        n=1,
        temperature=0.7,
        max_tokens=100,
    )

    print(response)

    assert len(response) == 2
    assert "germany" in response[0].country.lower()
    assert "berlin" in response[0].capital.lower()
    assert "france" in response[1].country.lower()
    assert "paris" in response[1].capital.lower()
