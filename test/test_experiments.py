from unittest.mock import patch

import pytest
from vectory.experiments import DatasetModel, Experiment


@patch("vectory.experiments.DatasetModel.get", side_effect=DatasetModel.DoesNotExist)
def test_add_experiment_dataset_not_found(dataset_get_mock):
    """Dataset does not exist"""
    with pytest.raises(DatasetModel.DoesNotExist):
        Experiment.create(
            name="ds_test",
            train_dataset="ex_test",
            model="cool_model",
            params={"lr": 0.001},
        )

    dataset_get_mock.assert_called_once()


def test_add_experiment_invalid_params():
    """Both params and params_path"""
    with pytest.raises(TypeError) as exc_info:
        Experiment.create(
            name="ex_test",
            train_dataset=DatasetModel(),
            model="cool_model",
            params={"lr": 0.001},
            params_path="hello",
        )

    assert "params" in str(exc_info.value)
    assert "param_path" in str(exc_info.value)


def test_add_experiment_params_not_found():
    """Params path doesnt exist"""
    with pytest.raises(FileNotFoundError):
        Experiment.create(
            name="ds_test",
            train_dataset=DatasetModel(),
            model="cool_model",
            params_path="something",
        )


@patch("vectory.experiments.ExperimentModel.create", return_value="ExperimentModel")
def test_add_experiment_success(experiment_mock):
    """Params"""
    experiment = Experiment.create(
        name="ds_test",
        train_dataset=DatasetModel(),
        model="cool_model",
        params={"hello": "hello"},
    )
    assert experiment.model == "ExperimentModel"


@patch("vectory.experiments.os.path.isfile", return_value=True)
@patch("vectory.experiments.open")
@patch("vectory.experiments.json.load", return_value={"key": "value"})
@patch("vectory.experiments.ExperimentModel.create", return_value="ExperimentModel")
def test_add_experiment_success_from_file(
    experiment_mock,
    json_load_mock,
    open_mock,
    isfile_mock,
):
    """Params_path"""
    experiment = Experiment.create(
        name="ds_test",
        train_dataset=DatasetModel(),
        model="cool_model",
        params_path="params_relative_path",
    )
    assert experiment.model == "ExperimentModel"
