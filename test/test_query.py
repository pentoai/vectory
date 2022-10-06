from unittest.mock import MagicMock, patch

import pytest
from vectory.db.models import DatasetModel, Query


@patch("vectory.db.models.DatasetModel.select", return_value=[])
def test_get_dataset_not_found(select_mock: MagicMock):
    """Dataset doesn't exist"""

    with pytest.raises(DatasetModel.DoesNotExist):
        Query(DatasetModel).get()

    select_mock.assert_called_once_with()


@patch("vectory.db.models.DatasetModel.select", return_value=["test", "test2"])
def test_get_datasets(select_mock: MagicMock):
    """Dataset doesn't exist"""

    datasets = Query(DatasetModel).get()

    assert len(datasets) == 2
    select_mock.assert_called_once_with()


@patch("vectory.db.models.os.path.abspath", return_value="test_path_absolute.csv")
def test_get_dataset_conditions(abspath_mock: MagicMock):
    """Get datasets"""
    query_mock = MagicMock()
    query_mock.where = MagicMock(return_value=query_mock)

    with patch(
        "vectory.db.models.DatasetModel.select", return_value=query_mock
    ) as select_mock:
        with pytest.raises(DatasetModel.DoesNotExist):
            Query(DatasetModel).get(name="test", csv_path="test_path_relative.csv")

        select_mock.assert_called_once_with()
        abspath_mock.assert_called_once_with("test_path_relative.csv")
        assert query_mock.where.call_count == 2

        expression_first_where = query_mock.where.call_args_list[0][0][0]
        assert expression_first_where.lhs.name == "name"
        assert expression_first_where.rhs == "test"

        expression_second_where = query_mock.where.call_args_list[1][0][0]
        assert expression_second_where.lhs.name == "csv_path"
        assert expression_second_where.rhs == "test_path_absolute.csv"
