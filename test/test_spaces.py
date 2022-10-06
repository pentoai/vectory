from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vectory.db.models import DatasetModel, ExperimentModel
from vectory.spaces import EmbeddingSpace


@patch("vectory.spaces.ExperimentModel.get", side_effect=ExperimentModel.DoesNotExist)
def test_add_embedding_space_experiment_not_found(experiment_get_mock: MagicMock):
    """Experiment does not exist"""
    with pytest.raises(ExperimentModel.DoesNotExist):
        EmbeddingSpace.create(
            name="embedding_space_test",
            npz_path="npz_path",
            dims=512,
            experiment="experiment_name",
            dataset="dataset_name",
        )

    experiment_get_mock.assert_called_once_with(name="experiment_name")


@patch("vectory.spaces.DatasetModel.get", side_effect=DatasetModel.DoesNotExist)
def test_add_embedding_space_dataset_not_found(dataset_get_mock: MagicMock):
    """Dataset does not exist"""
    with pytest.raises(DatasetModel.DoesNotExist):
        EmbeddingSpace.create(
            name="embedding_space_test",
            npz_path="npz_path",
            dims=512,
            experiment=ExperimentModel(),
            dataset="dataset_name",
        )

    dataset_get_mock.assert_called_once_with(name="dataset_name")


@patch("vectory.spaces.os.path.abspath", return_value="npz_absolute_path")
@patch("vectory.spaces.os.path.isfile", return_value=False)
def test_add_embedding_space_npz_not_found(isfile_mock, abspath_mock):
    """npz file doesnt exist does not exist"""
    with pytest.raises(FileNotFoundError):
        EmbeddingSpace.create(
            name="es_test",
            npz_path="npz_path",
            dims=512,
            experiment=ExperimentModel(),
            dataset=DatasetModel(),
        )


@patch("vectory.spaces.os.path.isfile", return_value=True)
@patch("vectory.spaces.load_embeddings_from_numpy", return_value=np.ones((2, 2)))
def test_add_embedding_space_incorrect_dims(
    isfile_mock,
    load_embeddings_mock,
):
    """Different dim shape"""
    with pytest.raises(ValueError):
        EmbeddingSpace.create(
            name="es_test",
            npz_path="npz_path",
            dims=512,
            experiment=ExperimentModel(),
            dataset=DatasetModel(),
        )


@patch("vectory.spaces.os.path.isfile", return_value=True)
@patch("vectory.spaces.load_embeddings_from_numpy", return_value=np.ones((2, 2)))
@patch("vectory.spaces.EmbeddingSpaceModel.create", return_value="EmbeddingSpaceModel")
def test_add_embedding_space_success(
    isfile_mock,
    load_embeddings_mock,
    embedding_space_mock,
):
    """Embedding Space"""
    embedding_space = EmbeddingSpace.create(
        name="es_test",
        npz_path="npz_path",
        dims=2,
        experiment=ExperimentModel(),
        dataset=DatasetModel(),
    )
    assert embedding_space.model == "EmbeddingSpaceModel"
