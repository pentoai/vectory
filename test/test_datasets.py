import os
from unittest.mock import MagicMock, patch

import pytest
from vectory.datasets import Dataset
from vectory.db.models import DatasetModel, ExperimentModel
from vectory.exceptions import IDFieldNotUnique, RecursiveOperationNotAllowedError
from vectory.experiments import Experiment

ROWS_repeated = [
    {"image_id": "n02979186_9036", "labels": "cassette_player,"},
    {"image_id": "n02979186_9036", "labels": "cassette_player,"},
]
ROWS = [
    {"image_id": "n02979186_9036", "labels": "cassette_player,"},
    {"image_id": "n1111111111111", "labels": "cassette_player,"},
]
HEADER = ["image_id", "labels"]
LOAD_CSV_WITH_HEADERS_REPEATED = (ROWS_repeated, "header mapping", HEADER)
LOAD_CSV_WITH_HEADERS = (ROWS, "header mapping", HEADER)


@pytest.fixture
def paths():
    csv_path = "CSV_PATH"
    csv_abspath = os.path.abspath(csv_path)

    return {"csv_path": csv_path, "csv_abspath": csv_abspath}


@patch(
    "vectory.datasets.load_csv_with_headers",
    side_effect=FileNotFoundError("File not found"),
)
def test_add_dataset_not_found_error(
    load_csv_with_headers: MagicMock,
    paths,
):
    """CSV path doesn't exist"""
    with pytest.raises(FileNotFoundError):
        Dataset.create(
            name="ds_test",
            csv_path=paths["csv_path"],
            id_field="image_id",
        )
    load_csv_with_headers.assert_called_once_with(paths["csv_abspath"], "image_id")


@patch(
    "vectory.datasets.load_csv_with_headers",
    return_value=LOAD_CSV_WITH_HEADERS_REPEATED,
)
def test_add_dataset_repeaded_ids(
    load_csv_with_headers: MagicMock,
    paths,
):
    """The CSV has repeated img_id"""
    with pytest.raises(IDFieldNotUnique):
        Dataset.create(
            name="ds_test",
            csv_path=paths["csv_path"],
            id_field="image_id",
        )

    load_csv_with_headers.assert_called_once_with(paths["csv_abspath"], "image_id")


@patch(
    "vectory.db.models.DatasetModel.create",
    autospec=True,
    return_value="DatasetModel",
)
@patch(
    "vectory.datasets.load_csv_with_headers",
    return_value=LOAD_CSV_WITH_HEADERS,
)
def test_add_dataset(
    load_csv_with_headers: MagicMock, dataset_create_mock: MagicMock, paths
):
    """Creates a Dataset without problems and stores it the database"""
    dataset = Dataset.create(
        name="ds_test", csv_path=paths["csv_path"], id_field="image_id"
    )

    load_csv_with_headers.assert_called_once_with(paths["csv_abspath"], "image_id")
    dataset_create_mock.assert_called_once_with(
        name="ds_test", csv_path=paths["csv_abspath"], id_field="image_id"
    )
    assert dataset.model == "DatasetModel"


def test_delete_dataset_no_recursion():
    """Recursive operation not allowed"""
    with pytest.raises(RecursiveOperationNotAllowedError):
        dataset: Dataset = Dataset(dataset=DatasetModel())
        dataset.model.get_experiments = MagicMock(return_value=[1, 2])
        dataset.model.get_embedding_spaces = MagicMock(return_value=[1, 2, 3])
        dataset.delete_instance(recursive=False)


@patch("vectory.datasets.ElasticKNNClient")
@patch("vectory.experiments.ElasticKNNClient")
def test_delete_dataset_with_recursion(
    dataset_elastic_knn_mock: MagicMock, experiment_elastic_knn_mock: MagicMock
):
    """Recursive operation allowed"""
    dataset = Dataset(dataset=DatasetModel())
    experiment1 = Experiment(experiment=ExperimentModel())
    dataset.model.get_experiments = MagicMock(return_value=[experiment1])
    dataset.model.get_embedding_spaces = MagicMock(return_value=["space_1", "space_2"])
    dataset.model.get_elasticsearch_indices = MagicMock(
        return_value=["index_1", "index_2"]
    )
    experiment1.model.get_embedding_spaces = MagicMock(return_value=["space_1"])
    experiment1.model.get_elasticsearch_indices = MagicMock(return_value=["index_1"])

    with patch(
        "vectory.datasets.DatasetModel.delete_instance"
    ) as dataset_delete_instance_mock, patch(
        "vectory.experiments.ExperimentModel.delete_instance"
    ) as experiment_delete_instance_mock:
        dataset.delete_instance(recursive=True)

        dataset.model.get_experiments.assert_called_once_with()
        dataset.model.get_embedding_spaces.assert_called_once_with()

        dataset.model.get_elasticsearch_indices.assert_called_once_with()
        dataset_delete_instance_mock.assert_called_once_with(recursive=True)
        dataset_elastic_knn_mock.assert_called_once_with()
        experiment_elastic_knn_mock.assert_called_once_with()
        experiment_delete_instance_mock.assert_called_once_with(
            recursive=True, delete_nullable=False
        )
