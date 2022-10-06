import os
from pathlib import Path
from typing import Optional, Union

from vectory.db.models import DatasetModel
from vectory.es.client import ElasticKNNClient
from vectory.es.utils import load_csv_with_headers
from vectory.exceptions import IDFieldNotUnique, RecursiveOperationNotAllowedError
from vectory.utils import generate_name


class Dataset:
    def __init__(self, dataset: DatasetModel) -> None:
        self.model = dataset

    @classmethod
    def get(cls, *args, **kwargs) -> "Dataset":
        dataset = DatasetModel.get(*args, **kwargs)

        return Dataset(dataset)

    @classmethod
    def get_or_create(
        cls,
        csv_path: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        id_field: str = "_idx",
        **kwargs,
    ) -> "Dataset":
        try:
            dataset = DatasetModel.get(name=name)

            return Dataset(dataset)
        except DatasetModel.DoesNotExist:
            if csv_path is None:
                raise ValueError("Cannot create a dataset without a CSV index")

            return Dataset.create(
                csv_path=csv_path, name=name, id_field=id_field, **kwargs
            )

    @classmethod
    def create(
        cls,
        csv_path: Union[str, Path],
        name: Optional[str] = None,
        id_field: str = "_idx",
        **kwargs,
    ) -> "Dataset":
        """Create a new DatasetModel record from a CSV file"""

        csv_abs_path = str(os.path.abspath(csv_path))
        rows, _, _ = load_csv_with_headers(csv_abs_path, id_field)

        ids = []
        for row in rows:
            ids.append(row[id_field])

        if len(ids) != len(set(ids)):
            raise IDFieldNotUnique("Error: the id field column must have unique values")

        if name is None:
            name = generate_name("dataset")

        dataset = DatasetModel.create(
            name=name, csv_path=csv_abs_path, id_field=id_field, **kwargs
        )

        return Dataset(dataset)

    def delete_instance(self, recursive: bool = False, **kwargs):
        # Raises DatasetModel.DoNotExist if can't find dataset with name equals `name`
        experiments = self.model.get_experiments()
        embedding_spaces = self.model.get_embedding_spaces()

        #
        # If there is any experiment or embedding space, but `recursive` is
        # False, then we stop the execution of this function
        #
        if (experiments or embedding_spaces) and not recursive:
            raise RecursiveOperationNotAllowedError(
                "Deleting this dataset will delete all the "
                "experiments and embedding spaces generated from it. "
                "If this is what you want, set recursive True."
            )

        #
        # If we also want to remove the ElasticSearch indexes associated with the
        # embedding spaces, we delete them here
        #
        if recursive:
            indices = self.model.get_elasticsearch_indices()

            with ElasticKNNClient() as es:
                es.delete_indices(indices)

            for experiment in experiments:
                experiment.delete_instance(recursive=recursive, **kwargs)

        return self.model.delete_instance(recursive=recursive, **kwargs)
