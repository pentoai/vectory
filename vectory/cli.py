from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import typer
from playhouse.shortcuts import model_to_dict
from tabulate import tabulate
from vectory.datasets import Dataset
from vectory.db.models import (
    BaseModel,
    DatasetModel,
    ElasticSearchIndexModel,
    EmbeddingSpaceModel,
    ExperimentModel,
    KNNBulkRelationship,
    Query,
    create_db_tables,
    database,
)
from vectory.demo import DemoDatasets, ModelInfo, download_demo_data, prepare_demo_data
from vectory.es import docker
from vectory.es.api import Mapping
from vectory.es.client import ElasticKNNClient
from vectory.experiments import Experiment
from vectory.indices import delete_index, list_indices, load_index
from vectory.spaces import EmbeddingSpace, compare_embedding_spaces
from vectory.utils import get_vectory_dir
from vectory.visualization.run import run_streamlit

create_db_tables()

MODELS_DB = {
    "dataset": DatasetModel,
    "experiment": ExperimentModel,
    "space": EmbeddingSpaceModel,
    "index": ElasticSearchIndexModel,
}

MODELS = {
    "dataset": Dataset,
    "experiment": Experiment,
    "space": EmbeddingSpace,
}

app = typer.Typer()
elastic_app = typer.Typer()


@elastic_app.command()
def up(
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Run elasticsearch in the background"
    )
):
    """Start elasticsearch inside a docker container."""
    typer.secho("Starting vectory's elasticsearch...", fg=typer.colors.GREEN)
    docker.up(detach=detach)


@elastic_app.command()
def down():
    """Stop elasticsearch."""
    typer.secho("Stopping vectory's elasticsearch...", fg=typer.colors.GREEN)
    docker.down()


app.add_typer(elastic_app, name="elastic")


@app.command()
def demo(
    dataset_name: DemoDatasets = typer.Argument(
        DemoDatasets.cv,
        help=(
            "Name of the dataset for the demo. Can be either 'tiny-imagenet-200' for a "
            "computer vision demo or 'imdb' for a nlp one. If none given, the computer "
            "vision demo will be chosen."
        ),
    ),
    data_path: Path = typer.Option(
        get_vectory_dir() / "demo",
        "--data-path",
        help="Path to the demo files",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    run: bool = typer.Option(
        True,
        "--run/--no-run",
        help=(
            "Run the visualizations after data has been prepared. "
            "A shortcut to `vectory run`"
        ),
    ),
):
    typer.secho("This might take a while...", fg=typer.colors.MAGENTA)
    data_path = data_path / dataset_name.value

    if dataset_name == DemoDatasets.cv:
        models_info = [
            ModelInfo(name="resnet50", dims=512, demodataset=DemoDatasets.cv),
            ModelInfo(name="convnext-tiny", dims=768, demodataset=DemoDatasets.cv),
        ]
    elif dataset_name == DemoDatasets.nlp:
        models_info = [
            ModelInfo(name="bert", dims=1024, demodataset="nlp"),
            ModelInfo(name="roberta", dims=768, demodataset="nlp"),
        ]

    download_demo_data(dataset_name.value, models_info, data_path)

    typer.secho("Starting vectory's elasticsearch...", fg=typer.colors.GREEN)
    docker.up(detach=True, wait=True)

    typer.secho(f"Loading {dataset_name.value} data", fg=typer.colors.GREEN)
    with database.atomic():
        prepare_demo_data(dataset_name.value, models_info, data_path)

    database.close()

    if run:
        typer.secho("Running visualizations Streamlit app", fg=typer.colors.GREEN)
        run_streamlit()


@app.command()
def add(
    npz_path: Path = typer.Option(
        ...,
        "--embeddings",
        "-e",
        help="Path to the npy or npz file to load embeddings to the embedding space",
        prompt=True,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    csv_path: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Path to the dataset CSV file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    params_path: Optional[Path] = typer.Option(
        None,
        "--params",
        "-p",
        help="Path to JSON with parameters used for a trained experiment",
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    dims: Optional[int] = typer.Option(
        None, help="Number of dimensions in the embedding space"
    ),
    dataset_name: Optional[str] = typer.Option(
        None,
        help=(
            "Name of the dataset to generate the embedding space from. If none given,"
            "a random name will be generated."
        ),
    ),
    experiment_name: Optional[str] = typer.Option(
        None, help="Name of the experiment to create"
    ),
    train_dataset: Optional[str] = typer.Option(
        None,
        help="Name of the dataset from which the experiment was trained."
        "If it is was not loaded, leave empty",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        help="Name of the model used for the experiment",
    ),
    embedding_space_name: Optional[str] = typer.Option(
        None, help="Name of the embedding space that will be created"
    ),
    id_field: str = typer.Option(
        "_idx",
        help=(
            "Name of the field to use as id for elasticsearch index. If none given,"
            "random ids will be generated with the name '_idx'"
        ),
    ),
    load: bool = typer.Option(
        False,
        "--load",
        help=("Load embedding space into elastic search, using default values"),
    ),
):
    """Create datasets, experiments, embedding spaces and elasticsearch indices"""

    if not dataset_name and csv_path:
        typer.confirm(
            "A random dataset name will be generated since no dataset-name was given. "
            "Do you want to continue?",
            abort=True,
        )

    if not experiment_name:
        typer.confirm(
            "A random experiment will be generated since no experiment-name was given. "
            "Do you want to continue?",
            abort=True,
        )

    if experiment_name and not train_dataset:
        train_dataset = typer.prompt(
            "Enter the name of the train dataset", default="", type=str
        )
        train_dataset = train_dataset if train_dataset else None

    with database.atomic() as trx:
        try:
            dataset = Dataset.get_or_create(
                name=dataset_name, csv_path=csv_path, id_field=id_field
            )

            experiment = Experiment.get_or_create(
                train_dataset=train_dataset,
                model=model_name,
                name=experiment_name,
                params_path=params_path,
            )

            embedding_space = EmbeddingSpace.create(
                npz_path=str(npz_path),
                dims=dims,
                experiment=experiment,
                dataset=dataset,
                name=embedding_space_name,
            )
        except Exception as ex:
            trx.rollback()
            typer.secho(
                f"Couldn't add dataset, experiment or embedding: {ex}", fg="red"
            )

            return

    if load:
        load_index(
            index_name=embedding_space.model.name,
            embedding_space_name=embedding_space.model.name,
        )

    typer.secho("Done", fg="green")


def _models_to_data(models: List[BaseModel]) -> Tuple[List[str], List[Any]]:
    models_data = [model_to_dict(model) for model in models]

    headers: List[str] = [key for key in models_data[0].keys() if key != "created"]

    values = []
    for model_data in models_data:
        row = []
        for key, value in model_data.items():
            if key == "created":
                continue

            if isinstance(value, dict) and "created" in value:
                row.append(value["name"])
            else:
                row.append(value)

        values.append(row)

    return headers, values


@app.command()
def ls():
    """List all created datasets, experiments, embedding spaces and indices"""
    for model_name, model in MODELS_DB.items():
        typer.secho(f"{model_name.capitalize()} rows", fg="yellow")
        models = Query(model).get(empty_ok=True)

        if models:
            headers, values = _models_to_data(models)
        else:
            values = []
            headers = ["No rows"]

        typer.echo(tabulate(values, headers=headers))
        typer.echo()


def _delete_model_instance(model_cls, name: str, recursive: bool):
    model_instance = model_cls.get(name=name)
    if recursive:
        delete = typer.confirm(
            f"Deleting {name} with recursive True will delete all related models"
            " and indices. Are you sure you want to continue?"
        )
        if not delete:
            raise typer.Abort()
    model_instance.delete_instance(recursive=recursive)
    typer.echo(f"{model_cls.__name__} object deleted")


dataset_app = typer.Typer()
experiment_app = typer.Typer()
embeddings_app = typer.Typer()


def _add_delete_command(group, model_cls):
    @group.command()
    def delete(name: str, recursive: bool = False):
        f"""Delete {model_cls.__name__} object"""
        _delete_model_instance(model_cls, name=name, recursive=recursive)


_add_delete_command(dataset_app, Dataset)
_add_delete_command(experiment_app, Experiment)
_add_delete_command(embeddings_app, EmbeddingSpace)

app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")
app.add_typer(embeddings_app, name="embeddings")


@app.command()
def compare(
    embedding_space_a: str = typer.Argument(
        ...,
        help="First embedding space to compare",
    ),
    embedding_space_b: str = typer.Argument(
        ...,
        help="Second embedding space to compare",
    ),
    metric_a: str = typer.Option(
        "euclidean",
        "-ma",
        help="Distance metric to use for kNN search in embedding space a",
    ),
    metric_b: str = typer.Option(
        "euclidean",
        "-mb",
        help="Distance metric to use for kNN search in embedding space b",
    ),
    precompute_knn: bool = typer.Option(
        False,
        "--precompute",
        help=(
            "Allow precompute knn when calculating jaccard similarity."
            "This operation may be expensive."
        ),
    ),
    calculate_histogram: bool = typer.Option(
        False, "--histogram", help="Calculate similarity histogram"
    ),
):
    """Compare two embedding spaces"""
    try:
        similarity, _, fig, _ = compare_embedding_spaces(
            embedding_space_a=embedding_space_a,
            embedding_space_b=embedding_space_b,
            metric_a=metric_a,
            metric_b=metric_b,
            allow_precompute_knn=precompute_knn,
            histogram=calculate_histogram,
        )
    except KNNBulkRelationship.DoesNotExist:
        typer.echo("Precompute knn not allowed.")
        raise typer.Abort()

    typer.echo(f"The mean of the Jaccard similarity for each query is {similarity}")
    if calculate_histogram:
        fig.tight_layout()
        plt.show()


@app.command()
def run():
    """Run the Streamlit app"""
    run_streamlit()


@embeddings_app.command()
def load(
    index_name: str = typer.Argument(
        ..., help="Name of the index to load embeddings into"
    ),
    embedding_space_name: str = typer.Argument(
        ..., help="Name of the embedding space to load"
    ),
    model: str = typer.Option(
        "lsh", help="Model to use for embedding kNN search [lsh, exact]:"
    ),
    similarity: str = typer.Option(
        "cosine", help="Similarity function to use for kNN search [cosine, l2]:"
    ),
    num_threads: int = typer.Option(
        4, "--num-threads", "-t", help="Number of threads to use"
    ),
    chunk_size: int = typer.Option(1000, "--chunk-size", "-c", help="Chunk size"),
    k: int = typer.Option(1, "-k", help="Hyperparameter for kNN search"),
    L: int = typer.Option(
        99, "-L", help="Hyperparameter for kNN search with 'lsh' model"
    ),
    w: int = typer.Option(
        3, "-w", help="Hyperparameter for kNN search with l2 smililarity"
    ),
    number_of_shards: int = typer.Option(
        1, "--num-shards", "-n", help="Number of shards to use"
    ),
):
    """Load the embeddings into elasticsearch"""
    es = ElasticKNNClient()

    if index_name in es.list_indices():
        typer.echo(f"Error: {index_name} is already loaded")
        es.close()
        raise typer.Abort()

    try:
        embedding_space = Query(EmbeddingSpaceModel).get(name=embedding_space_name)[0]
    except Exception:
        es.close()
        raise

    if model == "lsh":
        if similarity == "cosine":
            mapping = Mapping.CosineLsh(dims=embedding_space.dims, L=L, k=k)
        elif similarity == "l2":
            mapping = Mapping.L2Lsh(dims=embedding_space.dims, L=L, k=k, w=w)
        else:
            typer.echo(f"Invalid similarity {similarity}, try cosine or l2")
            raise typer.Abort()
    elif model == "exact":
        mapping = Mapping.DenseFloat(dims=embedding_space.dims)
    else:
        typer.echo(f"Invalid model {model}, try lsh or exact")
        es.close()
        raise typer.Abort()
    successes = load_index(
        index_name=index_name,
        embedding_space_name=embedding_space_name,
        mapping=mapping,
        chunk_size=chunk_size,
        num_threads=num_threads,
    )

    typer.echo(f"Finished loading {successes} embeddings!")


@embeddings_app.command("delete-index")
def delete_index_cmd(
    index_name: str = typer.Argument(..., help="Name of the index to delete"),
):
    """Delete an index"""
    result = delete_index(index_name=index_name)
    typer.echo(f"Index {index_name} deleted")
    typer.echo(result)


@embeddings_app.command()
def delete_all_indices():
    """Delete all indices"""
    confirmation = typer.confirm("Are you sure you want to delete all the indices?")

    if not confirmation:
        typer.echo("Not deleting")
        raise typer.Abort()

    es = ElasticKNNClient()
    for index_name in es.list_indices():
        if index_name != ".geoip_databases":
            delete_index(index_name=index_name)
            typer.echo(f"Index {index_name} deleted")

    typer.echo("All indices deletes!")
    es.close()


@embeddings_app.command("list-indices")
def list_indices_cmd():
    """List all indices"""
    typer.echo("")
    for index in list_indices():
        typer.echo(index)
    typer.echo("")
