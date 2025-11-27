import nltk
import pytest


@pytest.fixture(scope="session", autouse=True)
def download_models():
    nltk.download("punkt")
