import datetime
import os
from typing import Generic, List, Set, Type, TypeVar

from peewee import CharField, DateTimeField, IntegerField, Model
from playhouse.sqlite_ext import ForeignKeyField, JSONField, SqliteExtDatabase
from vectory.utils import get_vectory_dir

DB_PATH = get_vectory_dir() / "main.db"


database = SqliteExtDatabase(
    DB_PATH,
    pragmas={
        "journal_mode": "off",
        "synchronous": 0,
        "cache_size": -1 * 100000,  # 64MB
        "locking_mode": "exclusive",
        "temp_store": "memory",
        "foreign_keys": 1,
    },
)


class BaseModel(Model):
    class Meta:
        database = database


def _get_elastic_search_indices(
    embedding_spaces: List["EmbeddingSpaceModel"],
) -> List["ElasticSearchIndexModel"]:
    indices: Set[ElasticSearchIndexModel] = set()

    for embedding_space in embedding_spaces:
        for index in embedding_space.get_elasticsearch_indices():
            indices.add(index)

    return list(indices)


class DatasetModel(BaseModel):
    class Meta:
        database = database
        table_name = "datasets"

    name = CharField(unique=True)
    created = DateTimeField(default=datetime.datetime.now)
    csv_path = CharField()
    id_field = CharField()

    def __str__(self):
        return self.name

    def get_experiments(self) -> List["ExperimentModel"]:
        experiments = ExperimentModel.select().where(
            ExperimentModel.train_dataset == self
        )

        return list(experiments)

    def get_embedding_spaces(self) -> List["EmbeddingSpaceModel"]:
        embedding_spaces = EmbeddingSpaceModel.select().where(
            EmbeddingSpaceModel.dataset == self
        )

        return list(embedding_spaces)

    def get_elasticsearch_indices(self) -> List["ElasticSearchIndexModel"]:
        embedding_spaces = self.get_embedding_spaces()
        return _get_elastic_search_indices(embedding_spaces)


class ExperimentModel(BaseModel):
    class Meta:
        database = database
        table_name = "experiments"

    name = CharField(unique=True)
    model = CharField()
    created = DateTimeField(default=datetime.datetime.now)
    params = JSONField()
    train_dataset = ForeignKeyField(
        DatasetModel,
        backref="experiments",
        on_delete="CASCADE",
        on_update="CASCADE",
        null=True,
    )

    def __str__(self):
        return self.name

    def get_embedding_spaces(
        self,
    ) -> List["EmbeddingSpaceModel"]:
        embedding_spaces = EmbeddingSpaceModel.select().where(
            EmbeddingSpaceModel.experiment == self
        )

        return list(embedding_spaces)

    def get_elasticsearch_indices(self) -> List["ElasticSearchIndexModel"]:
        embedding_spaces = self.get_embedding_spaces()
        return _get_elastic_search_indices(embedding_spaces)


class EmbeddingSpaceModel(BaseModel):
    class Meta:
        database = database
        table_name = "embedding_spaces"

    name = CharField(unique=True)
    created = DateTimeField(default=datetime.datetime.now)
    npz_path = CharField()
    dims = IntegerField()
    experiment = ForeignKeyField(
        ExperimentModel,
        backref="embedding_spaces",
        on_delete="CASCADE",
        on_update="CASCADE",
    )
    dataset = ForeignKeyField(
        DatasetModel,
        backref="embedding_spaces",
        on_delete="CASCADE",
        on_update="CASCADE",
    )

    def __str__(self):
        return self.name

    def get_elasticsearch_indices(self) -> List["ElasticSearchIndexModel"]:
        return list(
            ElasticSearchIndexModel.select().where(
                ElasticSearchIndexModel.embedding_space == self
            )
        )


class ElasticSearchIndexModel(BaseModel):
    name = CharField(unique=True)
    embedding_space = ForeignKeyField(
        EmbeddingSpaceModel,
        backref="elasticsearch_indices",
        on_delete="CASCADE",
        on_update="CASCADE",
    )
    model = CharField()
    similarity = CharField(null=True)
    num_threads = IntegerField()
    chunk_size = IntegerField()
    k = IntegerField(null=True)
    L = IntegerField(null=True)
    w = IntegerField(null=True)
    number_of_shards = IntegerField()

    def __str__(self):
        return self.name


class KNNBulkRelationship(BaseModel):
    embedding_space = ForeignKeyField(
        EmbeddingSpaceModel,
        backref="knn_bulk_relationships",
        on_delete="CASCADE",
        on_update="CASCADE",
    )
    metric = CharField()

    class Meta:
        indexes = ((("embedding_space", "metric"), True),)


class KNNBulkStorage(BaseModel):
    relationship = ForeignKeyField(
        KNNBulkRelationship,
        backref="knn_bulk_storage",
        on_delete="CASCADE",
        on_update="CASCADE",
    )
    created = DateTimeField(default=datetime.datetime.now)
    npz_index = IntegerField()
    k_1 = IntegerField()
    k_2 = IntegerField()
    k_3 = IntegerField()
    k_4 = IntegerField()
    k_5 = IntegerField()
    k_6 = IntegerField()
    k_7 = IntegerField()
    k_8 = IntegerField()
    k_9 = IntegerField()
    k_10 = IntegerField()

    @staticmethod
    def get_knn(embedding_space_name: str, metric: str):
        return (
            KNNBulkStorage.select(
                KNNBulkStorage.k_1,
                KNNBulkStorage.k_2,
                KNNBulkStorage.k_3,
                KNNBulkStorage.k_4,
                KNNBulkStorage.k_5,
                KNNBulkStorage.k_6,
                KNNBulkStorage.k_7,
                KNNBulkStorage.k_8,
                KNNBulkStorage.k_9,
                KNNBulkStorage.k_10,
            )
            .join(KNNBulkRelationship)
            .where(KNNBulkRelationship.metric == metric)
            .join(EmbeddingSpaceModel)
            .where(EmbeddingSpaceModel.name == embedding_space_name)
            .order_by(KNNBulkStorage.npz_index)
        )


def create_db_tables():
    if not DB_PATH.parent.exists():
        os.makedirs(DB_PATH.parent, exist_ok=True)
        with database:
            database.create_tables(
                [
                    ExperimentModel,
                    DatasetModel,
                    EmbeddingSpaceModel,
                    ElasticSearchIndexModel,
                    KNNBulkRelationship,
                    KNNBulkStorage,
                ]
            )


Q = TypeVar(
    "Q", DatasetModel, ExperimentModel, EmbeddingSpaceModel, ElasticSearchIndexModel
)


class Query(Generic[Q]):
    """A class to query database models without many complications"""

    def __init__(self, model: Type[Q]) -> None:
        self.model: Type[Q] = model

    def get(self, empty_ok=False, **condition) -> List[Q]:
        # Replace path columns with absolute paths
        for key, value in condition.items():
            if "_path" in key:
                condition[key] = os.path.abspath(condition[key])

        condition = {k: v for k, v in condition.items() if v is not None}

        query = self.model.select()
        for column, value in condition.items():
            query = query.where(getattr(self.model, column) == value)

        results = list(query)

        if not empty_ok and not results:
            raise self.model.DoesNotExist

        return results
