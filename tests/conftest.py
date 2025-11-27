import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
