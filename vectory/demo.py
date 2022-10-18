import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Union

import requests  # type: ignore
import typer
from vectory.datasets import Dataset
from vectory.db.models import EmbeddingSpaceModel
from vectory.experiments import Experiment
from vectory.indices import load_index
from vectory.spaces import EmbeddingSpace, compare_embedding_spaces


@dataclass
class ModelInfo:
    name: str
    dims: int
    demodataset: str


class DemoDatasets(str, Enum):
    cv = "tiny-imagenet-200"
    nlp = "imdb"


def download_demo_data(
    dataset_name: str, models_info: List[ModelInfo], data_path: Union[str, Path]
) -> None:

    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

    base_url = "https://github.com/pentoai/vectory/releases/download/v0.1.0/"

    # Download dataset
    csv_name = f"{dataset_name}-data.csv"
    if not os.path.isfile(os.path.join(data_path, csv_name)):
        typer.secho("Downloading dataset", fg="yellow")
        r = requests.get(base_url + csv_name)
        with open(os.path.join(data_path, csv_name), "wb") as csv_file:
            csv_file.write(r.content)
            csv_file.close()
    else:
        typer.secho(f"Dataset {csv_name} already downloaded", fg="yellow")

    for model_info in models_info:
        # Download experiment
        if not os.path.isfile(os.path.join(data_path, f"{model_info.name}.npy")):
            typer.secho(
                f"Downloading {model_info.name} generated embeddings", fg="yellow"
            )
            r = requests.get(base_url + model_info.name + ".npy", stream=True)
            with open(
                os.path.join(data_path, f"{model_info.name}.npy"), "wb"
            ) as npx_file:
                npx_file.write(r.content)
                npx_file.close()
        else:
            typer.secho(
                f"Embeddings from {model_info.name} already downloaded", fg="yellow"
            )

    typer.secho("Done downloading demo data", fg="green")


def prepare_demo_data(
    dataset_name: str, models_info: List[ModelInfo], data_path: Union[str, Path]
) -> None:

    dataset = Dataset.get_or_create(
        name=dataset_name,
        csv_path=str(Path(data_path) / f"{dataset_name}-data.csv"),
        id_field="_idx",
    )
    typer.secho(f"Dataset {dataset_name} created", fg="yellow")

    for model_info in models_info:
        experiment = Experiment.get_or_create(
            train_dataset=dataset,
            model=model_info.name,
            name=model_info.name,
            params=None,
        )
        typer.secho(f"Experiment {model_info.name} created", fg="yellow")

        EmbeddingSpace.get_or_create(
            npz_path=str(os.path.join(data_path, f"{model_info.name}.npy")),
            dims=model_info.dims,
            experiment=experiment,
            dataset=dataset,
            name=model_info.name,
        )

        typer.secho(f"Embedding space {model_info.name} created", fg="yellow")

        try:
            index_name = model_info.demodataset + model_info.name
            load_index(
                index_name=index_name,
                embedding_space_name=model_info.name,
            )
            typer.secho(f"Created index {index_name}", fg="yellow")
        except ValueError:
            typer.secho(f"Index {model_info.name} already exists", fg="yellow")

    embedding_space_a = EmbeddingSpaceModel.get(
        EmbeddingSpaceModel.name == models_info[0].name
    )
    embedding_space_b = EmbeddingSpaceModel.get(
        EmbeddingSpaceModel.name == models_info[1].name
    )
    try:
        typer.secho(
            "Comparing embedding spaces, this might take a while...", fg="yellow"
        )
        compare_embedding_spaces(
            embedding_space_a.name, embedding_space_b.name, allow_precompute_knn=True
        )
        typer.secho(
            f"Compared {embedding_space_a.name} and {embedding_space_b.name}",
            fg="yellow",
        )
    except Exception:
        typer.secho("Couldn't compare embeddings", fg="red")
    typer.secho("Done preparing demo", fg="green")
