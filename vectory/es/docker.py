import subprocess
import sys
from functools import wraps
from pathlib import Path
from time import sleep, time
from typing import List

import docker
from rich.console import Console

err_console = Console(stderr=True)
COMPOSE_FILE = Path(__file__).parent.parent.parent / "docker-compose.yml"
CONTAINER_NAME = "es-embeddings-db"


def get_container_health(container_name: str) -> str:
    api_client = docker.APIClient()  # type: ignore
    inspect_results = api_client.inspect_container(container_name)

    return inspect_results["State"]["Health"]["Status"]


def wait_until_healthy(container_name: str, timeout: int = 60):
    start_time = time()
    while get_container_health(container_name).lower() != "healthy":
        if time() - start_time > timeout:
            raise Exception(f"Failed to validate {container_name} container health")

        sleep(1)


def docker_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except subprocess.SubprocessError as ex:
            err_console.print(
                f"`vectory elastic {func.__name__}` requires docker to be running. "
                "Is docker installed and running?\n\n"
                f"Error: {ex}"
            )

    return wrapper


@docker_handler
def up(detach: bool = True, wait: bool = False):
    """Start elasticsearch inside a docker container."""
    command: List[str] = ["docker-compose", "-f", str(COMPOSE_FILE.absolute()), "up"]

    if detach:
        command.append("-d")

    subprocess.run(
        command,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )

    if wait:
        wait_until_healthy(CONTAINER_NAME)


@docker_handler
def down():
    """Stop elasticsearch."""
    command: List[str] = ["docker-compose", "-f", str(COMPOSE_FILE.absolute()), "down"]

    subprocess.run(
        command,
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
