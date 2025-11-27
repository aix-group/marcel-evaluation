import asyncio
from typing import Literal, Optional, Type, TypeVar, overload

from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

T = TypeVar("T", bound=BaseModel)


class GenerationError(Exception):
    pass


_client = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if not _client:
        _client = AsyncOpenAI(max_retries=2)
    return _client


async def generate(
    messages,
    model: str,
    response_format: Type[T],
    n: int,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> list[T]:
    completions = []
    client = get_openai_client()

    max_attempts = 3
    attempts_left = max_attempts
    while attempts_left > 0:
        try:
            response = await client.chat.completions.parse(
                messages=messages,
                model=model,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            completions = [
                choice.message.parsed
                for choice in response.choices
                if choice.message.parsed is not None
            ]

            if len(completions) == n:
                break

        except (LengthFinishReasonError, ValidationError):
            pass  # a retry can recover from these errors

        attempts_left -= 1

    if len(completions) != n:
        raise GenerationError(
            f"Could not generate response for conversation: {messages} with response_format: {response_format}"
        )

    return completions


@overload
async def batch_generate(
    conversations,
    model: str,
    response_format: Type[T],
    n: Literal[1] = 1,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_concurrency: int = 1,
) -> list[T]: ...


@overload
async def batch_generate(
    conversations,
    model: str,
    response_format: Type[T],
    n: int,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_concurrency: int = 1,
) -> list[list[T]]: ...


async def batch_generate(
    conversations,
    model: str,
    response_format: Type[T],
    n: int = 1,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_concurrency: int = 1,
) -> list[T] | list[list[T]]:
    semaphore = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(conversations))

    async def call_generate(**kwargs):
        async with semaphore:
            result = await generate(**kwargs)
            pbar.update(1)
            if n == 1:
                return result[0]
            return result

    tasks = [
        asyncio.create_task(
            call_generate(
                messages=conversation,
                model=model,
                response_format=response_format,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        for conversation in conversations
    ]

    return await asyncio.gather(*tasks)
