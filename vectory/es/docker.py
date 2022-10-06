from functools import wraps
from pathlib import Path

import typer
from python_on_whales import DockerClient
from python_on_whales.exceptions import DockerException
from rich.console import Console

app = typer.Typer()
err_console = Console(stderr=True)

docker = DockerClient(
    compose_files=[Path(__file__).parent.parent.parent / "docker-compose.yml"]
)


def docker_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except DockerException:
            err_console.print(
                f"`vectory elastic {func.__name__}` requires docker to be running. "
                "Is docker installed and running?"
            )

    return wrapper


@app.command()
@docker_handler
def up(
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Run elasticsearch in the background"
    )
):
    """Start elasticsearch inside a docker container."""
    typer.secho("Starting vectory's elasticsearch...", fg=typer.colors.GREEN)
    docker.compose.up(detach=detach)


@app.command()
@docker_handler
def down():
    """Stop elasticsearch."""
    typer.secho("Stopping vectory's elasticsearch...", fg=typer.colors.GREEN)
    docker.compose.down()
