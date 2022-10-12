import json
import os
from pathlib import Path
from typing import Optional, Union

from vectory.datasets import Dataset
from vectory.db.models import DatasetModel, ExperimentModel
from vectory.es.client import ElasticKNNClient
from vectory.exceptions import RecursiveOperationNotAllowedError
from vectory.utils import generate_name


class Experiment:
    def __init__(self, experiment: ExperimentModel) -> None:
        self.model = experiment

    @classmethod
    def get(cls, *args, **kwargs) -> "Experiment":
        experiment = ExperimentModel.get(*args, **kwargs)

        return Experiment(experiment)

    @classmethod
    def get_or_create(
        cls,
        train_dataset: Optional[Union[str, DatasetModel, Dataset]] = None,
        model: Optional[str] = None,
        name: Optional[str] = None,
        params: Optional[Union[dict, str]] = None,
        params_path: Optional[Union[str, Path]] = None,
    ) -> "Experiment":
        try:
            experiment = ExperimentModel.get(name=name)
            return Experiment(experiment)
        except ExperimentModel.DoesNotExist:
            return Experiment.create(
                train_dataset=train_dataset,
                model=model,
                name=name,
                params=params,
                params_path=params_path,
            )

    @classmethod
    def create(
        cls,
        train_dataset: Optional[Union[str, DatasetModel, Dataset]] = None,
        model: Optional[str] = None,
        name: Optional[str] = None,
        params: Optional[Union[dict, str]] = None,
        params_path: Optional[Union[str, Path]] = None,
    ) -> ExperimentModel:
        if params_path is not None and params is not None:
            raise TypeError("Only one of 'params' and 'param_path' should be given")

        if params_path is not None:
            with open(os.path.abspath(params_path)) as f:
                params_json = json.load(f)
        elif params is not None:
            params_json = json.dumps(params)
        elif params is None:
            params_json = ""

        if isinstance(train_dataset, str):
            train_dataset = DatasetModel.get(name=train_dataset)
        elif isinstance(train_dataset, Dataset):
            train_dataset = train_dataset.model

        if name is None:
            name = generate_name("experiment")

        if model is None:
            model = "NA"

        experiment = ExperimentModel.create(
            name=name,
            model=model,
            train_dataset=train_dataset,
            params=params_json,
        )

        return Experiment(experiment)

    def delete_instance(self, recursive: bool = True):
        embedding_spaces = self.model.get_embedding_spaces()

        if embedding_spaces and not recursive:
            raise RecursiveOperationNotAllowedError(
                "Deleting this experiment will delete all the "
                "embedding spaces generated from it. If this is what you "
                "want, set recursive True."
            )

        if recursive:
            indices = self.model.get_elasticsearch_indices()

            with ElasticKNNClient() as es:
                es.delete_indices(indices)

        return self.model.delete_instance(recursive=recursive, delete_nullable=False)
