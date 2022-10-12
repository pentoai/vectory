from pathlib import Path

from coolname import generate


def generate_name(name: str, words=3) -> str:
    return "-".join([*generate(words), name])


def get_vectory_dir() -> Path:
    return Path.home() / ".vectory"
