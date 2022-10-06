from coolname import generate


def generate_name(name: str, words=3) -> str:
    return "-".join([*generate(words), name])
