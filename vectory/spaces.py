import os
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from vectory.datasets import Dataset
from vectory.db.bulk import KNNBulkOperations, KNNBulkRelationship
from vectory.db.models import DatasetModel, EmbeddingSpaceModel, ExperimentModel
from vectory.es.client import ElasticKNNClient
from vectory.es.utils import load_embeddings_from_numpy
from vectory.exceptions import OperationNotAllowedError
from vectory.experiments import Experiment
from vectory.utils import generate_name


class EmbeddingSpace:
    def __init__(self, embedding_space: EmbeddingSpaceModel) -> None:
        self.model = embedding_space

    @classmethod
    def get(cls, *args, **kwargs) -> "EmbeddingSpace":
        embedding_space = EmbeddingSpaceModel.get(*args, **kwargs)

        return EmbeddingSpace(embedding_space)

    @classmethod
    def get_or_create(
        cls,
        npz_path: str,
        experiment: Union[str, ExperimentModel],
        dataset: Union[str, DatasetModel],
        dims: Optional[int] = None,
        name: Optional[str] = None,
    ) -> "EmbeddingSpace":
        try:
            embedding_space = EmbeddingSpaceModel.get(name=name)
            return EmbeddingSpace(embedding_space)
        except EmbeddingSpaceModel.DoesNotExist:
            return EmbeddingSpace.create(
                npz_path=npz_path,
                experiment=experiment,
                dataset=dataset,
                dims=dims,
                name=name,
            )

    @classmethod
    def create(
        cls,
        npz_path: str,
        experiment: Union[str, ExperimentModel, Experiment],
        dataset: Union[str, DatasetModel, Dataset],
        dims: Optional[int] = None,
        name: Optional[str] = None,
    ):
        if isinstance(experiment, str):
            experiment = ExperimentModel.get(name=experiment)
        elif isinstance(experiment, Experiment):
            experiment = experiment.model

        if isinstance(dataset, str):
            dataset = DatasetModel.get(name=dataset)
        elif isinstance(dataset, Dataset):
            dataset = dataset.model

        npz_abs_path = os.path.abspath(npz_path)
        if not os.path.isfile(npz_abs_path):
            raise FileNotFoundError(f"Could not find file {npz_abs_path}")

        embeddings = load_embeddings_from_numpy(npz_abs_path)
        if dims is None:
            dims = embeddings.shape[1]
        elif embeddings.shape[1] != dims:
            raise ValueError("The given dimensions doesn't match npz dimensions")

        if name is None:
            name = generate_name("embedding-space")

        embedding_space = EmbeddingSpaceModel.create(
            name=name,
            npz_path=npz_abs_path,
            dims=dims,
            experiment=experiment,
            dataset=dataset,
        )

        return EmbeddingSpace(embedding_space)

    def delete_instance(self, recursive: bool = True):
        indices = self.model.get_elasticsearch_indices()

        if indices and recursive:
            with ElasticKNNClient() as es:
                es.delete_indices(indices)

        return self.model.delete_instance(recursive=recursive, delete_nullable=False)


def precompute_knn(
    embedding_space: str, metric: str, allow_precompute_knn: bool
) -> None:
    """Precompute KNN for the given embedding space."""
    try:
        KNNBulkRelationship.get(
            embedding_space=EmbeddingSpaceModel.get(name=embedding_space),
            metric=metric,
        )
    except KNNBulkRelationship.DoesNotExist:
        if allow_precompute_knn:
            KNNBulkOperations(embedding_space_name=embedding_space).index(metric=metric)
        else:
            raise OperationNotAllowedError(
                "Expensive operation not allowed. Cannot precompute KNN."
            )


def compare_embedding_spaces(
    embedding_space_a: str,
    embedding_space_b: str,
    metric_a: str = "euclidean",
    metric_b: str = "euclidean",
    allow_precompute_knn: bool = False,
    histogram: bool = False,
) -> Tuple[np.ndarray, Dict[str, float], plt.Figure, plt.Axes]:
    """
    Compute the similarity between two embedding spaces using the given metrics.

    Parameters
    ----------
    embedding_space_a : str
        The name of the first embedding space.
    embedding_space_b : str
        The name of the second embedding space.
    metric_a : str, optional
        The metric to use for the first embedding space, by default "euclidean"
    metric_b : str, optional
        The metric to use for the second embedding space, by default "euclidean"
    allow_precompute_knn : bool, optional
        Whether to allow precomputing the KNN for the given embedding spaces, by
        default False
    histogram : bool, optional
        Whether to plot a histogram of the similarities, by default False

    Returns
    -------
    Tuple(np.ndarray, Dict[str, float], plt.Figure, plt.Axes)
        The similarities, the similarity statistics, the figure and the axes.
    """

    precompute_knn(embedding_space_a, metric_a, allow_precompute_knn)
    precompute_knn(embedding_space_b, metric_b, allow_precompute_knn)

    try:
        id_similarity_dict = KNNBulkOperations.space_similarity(
            embedding_space_name_a=embedding_space_a,
            metric_a=metric_a,
            embedding_space_name_b=embedding_space_b,
            metric_b=metric_b,
        )
    except AssertionError:
        raise AssertionError(
            "Cannot compute similarity, please calculate the space similarity first."
        )

    spaces_similarity = np.array(list(id_similarity_dict.values())).mean()

    fig, ax = (None, None)
    if histogram:
        similarity = np.array(list(id_similarity_dict.values()))

        density, bin_edges = np.histogram(
            similarity, np.arange(0.0, 1.1, 0.1), normed=True, density=True
        )
        unity_density = density / density.sum()
        bins = [
            f"{bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}"
            for i in range(len(bin_edges) - 1)
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(bins, unity_density)
        ax.set_xlabel("IoU")
        ax.set_ylabel("Density")
        fig.tight_layout()

    return spaces_similarity, id_similarity_dict, fig, ax
