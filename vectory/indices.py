import random
from typing import List

import typer
from elasticsearch import NotFoundError
from vectory.db.models import ElasticSearchIndexModel, EmbeddingSpaceModel
from vectory.es.api import Mapping, NearestNeighborsQuery, Similarity, Vec
from vectory.es.client import ElasticKNNClient
from vectory.es.utils import load_csv_with_headers, load_embeddings_from_numpy
from vectory.exceptions import CreatingIndexError, DimensionsDismatchError


def load_index(
    index_name: str,
    embedding_space_name: str,
    mapping: Mapping = None,
    num_threads: int = 4,
    chunk_size: int = 1000,
    number_of_shards: int = 1,
) -> int:
    """
    Load an index into ElasticSearch.

    Parameters
    ----------
    index_name : str
        The name that is going to be given to the index.
    embedding_space_name : str
        The name of the embedding space which's embeddings are going to be loaded.
    mapping : Mapping, optional
        The mapping that is going to be used for the index. If not given, the CosineLsh
        mapping will be used. See `vectory.es.api.Mapping` for more information.
    num_threads : int, optional
        The number of threads that are going to be used for the index creation. Defaults
        to 4.
    chunk_size : int, optional
        The number of embeddings that are going to be loaded at once. Defaults to 1000.
    number_of_shards : int, optional
        The number of shards that are going to be used for the index. Defaults to 1.

    Returns
    -------
    int
        The number of embeddings that were loaded into the index.
    """

    es = ElasticKNNClient()

    if index_name in es.list_indices():
        es.close()
        raise ValueError(f"Error: {index_name} is already loaded")

    embedding_space = EmbeddingSpaceModel.get(name=embedding_space_name)

    embeddings = load_embeddings_from_numpy(embedding_space.npz_path)
    rows, header_mapping, header = load_csv_with_headers(
        embedding_space.dataset.csv_path
    )

    if mapping is None:
        mapping = Mapping.CosineLsh(dims=embedding_space.dims, L=99, k=1)

    if embeddings.shape[0] != len(rows):
        es.close()
        raise DimensionsDismatchError("csv and npz have different number of rows")
    if embeddings.shape[1] != embedding_space.dims:
        es.close()
        raise DimensionsDismatchError("embedding_space.dims does not match npz dims")
    if embedding_space.dims != mapping.dims:
        es.close()
        raise DimensionsDismatchError("mapping.dims does not match npz dims")

    if isinstance(mapping, Mapping.CosineLsh):
        similarity = "cosine"
        model = "lsh"
        k = mapping.k
        L = mapping.L
        w = None
    elif isinstance(mapping, Mapping.L2Lsh):
        similarity = "l2"
        model = "lsh"
        k = mapping.k
        L = mapping.L
        w = mapping.w
    elif isinstance(mapping, Mapping.DenseFloat):
        similarity = None
        model = "exact"
        k = None
        L = None
        w = None
    else:
        es.close()
        raise ValueError("'Mapping' not suported")

    if len(rows) != len(embeddings):
        es.close()
        raise DimensionsDismatchError(
            "Error: number of rows in csv file doesn't match the number of embeddings"
            "in npz file"
        )

    typer.echo(
        f"Creating index {index_name} with similarity {similarity} and model {model}"
    )

    response = es.create_index(
        index_name=index_name,
        mapping=mapping,
        header_mapping=header_mapping,
        number_of_shards=number_of_shards,
    )

    if not response["acknowledged"]:
        es.close()
        raise CreatingIndexError("An error occurred while creating the index.")

    try:
        successes = es.index(
            index=index_name,
            embeddings=embeddings,
            metadata=rows,
            id_field=embedding_space.dataset.id_field,
            num_threads=num_threads,
            chunk_size=chunk_size,
        )
    except Exception:
        es.delete_index(index_name=index_name)
        es.close()
        raise MemoryError(
            "Couldn't load indices to Elasticsearch due to lack of memory"
        )
    es.close()

    assert successes == len(rows)
    try:
        ElasticSearchIndexModel.create(
            name=index_name,
            embedding_space=embedding_space,
            model=model,
            similarity=similarity,
            num_threads=num_threads,
            chunk_size=chunk_size,
            k=k,
            L=L,
            w=w,
            number_of_shards=number_of_shards,
        )
    except Exception as e:
        delete_index(index_name=index_name)
        es.close()
        raise e

    print("Index loaded")
    return successes


def delete_index(index_name: str):
    try:
        es_index = ElasticSearchIndexModel.get(name=index_name)
    except ElasticSearchIndexModel.DoesNotExist:
        raise ValueError(f"Error: `{index_name}` is not an index")

    es = ElasticKNNClient()

    if index_name not in es.list_indices():
        raise ValueError(f"Error: {index_name} is not an index")
    else:
        result = es.delete_index(index_name=index_name)
        es_index.delete_instance(recursive=True, delete_nullable=False)

    es.close()

    return result


def list_indices():
    es = ElasticKNNClient()
    indices = es.list_indices()
    es.close()

    return indices


def match_query(
    indices_name: List[str],
    query_id: str = None,
    similarity: str = "cosine",
    random_query: bool = False,
    k: int = 10,
    fetch_source: bool = False,
):
    if (query_id is None) and (not random_query):
        raise ValueError("Either 'query_id' or 'random_query' must be given.")

    # Check if all indices have the same dataset
    dataset = ElasticSearchIndexModel.get(name=indices_name[0]).embedding_space.dataset

    for index_name in indices_name:
        try:
            index = ElasticSearchIndexModel.get(name=index_name)
        except ElasticSearchIndexModel.DoesNotExist:
            raise ValueError(f"{index_name} is not an index")

        if index.embedding_space.dataset.name != dataset.name:
            raise ValueError(  # CHANGE
                f"Error: All indices must have the same dataset, but index "
                f"{index_name} has dataset {index.embedding_space.dataset.name} "
                f"and index {indices_name[0]} has dataset {dataset.name}"
            )
    es = ElasticKNNClient()

    if random_query:
        rows, _, header = load_csv_with_headers(dataset.csv_path)  # type: ignore
        query_id = random.choice(rows)[dataset.id_field]

    results = {}
    print(f"Querying: {dataset.id_field} = {query_id}")
    for index_name in indices_name:
        try:

            index = ElasticSearchIndexModel.get(name=index_name)
            query_vec = Vec.Indexed(index=index_name, id=query_id)
            if index.model == "lsh":
                if index.similarity == "cosine":
                    query = NearestNeighborsQuery.CosineLsh(vec=query_vec)
                elif index.similarity == "l2":
                    query = NearestNeighborsQuery.L2Lsh(vec=query_vec)
            elif index.model == "exact":
                if similarity == "cosine":
                    query = NearestNeighborsQuery.Exact(
                        vec=query_vec, similarity=Similarity.Cosine
                    )
                if similarity == "l2":
                    query = NearestNeighborsQuery.Exact(
                        vec=query_vec, similarity=Similarity.L2
                    )

            knn_neighbors = es.nearest_neighbors(
                index=index_name,
                query=query,
                id_field=dataset.id_field,
                k=k,
                fetch_source=fetch_source,
            )
            knn_result = []
            for result in knn_neighbors["hits"]["hits"]:
                knn_result.append(
                    (result["fields"][f"{dataset.id_field}"][0], result["_score"])
                )
            results[f"{index_name}"] = knn_result

        except NotFoundError:
            es.close()
            raise ValueError(
                f"Error: `{query_id}` did not match any value for the "
                f"keyword `{index.embedding_space.dataset.id_field}` at the"
                f" index `{index.name}`"
            )
    es.close()

    return results, query_id
