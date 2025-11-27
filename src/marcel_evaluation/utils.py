import json
from typing import List

from w3lib.url import canonicalize_url

from marcel_evaluation.base import Response


def clean_url(u):
    u = canonicalize_url(u)
    if u.startswith("http://"):
        u = u[7:]
    if u.startswith("https://"):
        u = u[8:]
    if u.startswith("www."):
        u = u[4:]
    if u.endswith("/"):
        u = u[:-1]
    return u


def load_run(json_path) -> List[Response]:
    with open(json_path) as fin:
        responses = json.load(fin)

    filtered = []
    for response in responses:
        # sanitize urls
        sources = [clean_url(url) for url in response["sources"]]
        contexts = []
        for context in response["contexts"] if response["contexts"] else []:
            context.copy()
            context["url"] = clean_url(context["url"])
            contexts.append(context)
        response.update({"contexts": contexts, "sources": sources})

        if not response["reference_answer"]:
            print(f"WARNING: {response['id']} has no reference answer.")

        if not response["generated_answer"]:
            print(f"WARNING: {response['id']} has no generated answer.")

        if isinstance(response["generated_answer"], str):
            response["generated_answer"] = [response["generated_answer"]]

        filtered.append(response)

    return filtered
