import resource
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import psutil
import umap
from bokeh.palettes import RdYlBu10
from sklearn.decomposition import PCA
from vectory.db.bulk import KNNBulkOperations
from vectory.db.models import ElasticSearchIndexModel, EmbeddingSpaceModel, Query
from vectory.es.utils import load_csv_with_headers, load_embeddings_from_numpy
from vectory.indices import match_query

GREEN = "#f5fff6"
RED = "#fadedc"


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * 0.95), hard))


def get_memory():
    available_memory = psutil.virtual_memory().available
    return available_memory


def umap_calc(embeddings):
    try:
        return umap.UMAP(n_components=2, n_jobs=-1).fit_transform(embeddings)
    except MemoryError:
        return umap.UMAP(n_components=2, n_jobs=-1, low_memory=True).fit_transform(
            embeddings
        )


def pca_calc(embeddings):
    return PCA(n_components=2).fit_transform(embeddings)


def pca_plus_umap_calc(embeddings):
    pca_points = PCA(n_components=50).fit_transform(embeddings)
    return umap_calc(pca_points)


def calculate_points(model, embeddings, rows):
    if embeddings.shape[1] == 2:
        dim_2_points = embeddings
    else:
        if model == "UMAP":
            dim_2_points = umap_calc(embeddings)
        elif model == "PCA":
            dim_2_points = pca_calc(embeddings)
        elif model == "PCA + UMAP":
            dim_2_points = pca_plus_umap_calc(embeddings)

    data = {}
    for rowl in rows:
        for key in rowl:
            aux = data.get(key, [])
            aux.append(rowl[key])
            data[key] = aux

    data["d1"] = dim_2_points[:, 0]
    data["d2"] = dim_2_points[:, 1]
    df = pd.DataFrame(data=data)

    return df


def format(index):
    if index is not None:
        return f"{index.name} ({index.model} ,{index.similarity})"
    return None


def color_positive_green(df):
    if df.coincidence:
        color = GREEN
    else:
        color = RED
    return [f"background-color: {color}"] * len(df)


def map_coincidence(val):
    if val <= 0.1:
        return "<0.1"
    elif val <= 0.2:
        return "<0.2"
    elif val <= 0.3:
        return "<0.3"
    elif val <= 0.4:
        return "<0.4"
    elif val <= 0.5:
        return "<0.5"
    elif val <= 0.6:
        return "<0.6"
    elif val <= 0.7:
        return "<0.7"
    elif val <= 0.8:
        return "<0.8"
    elif val <= 0.9:
        return "<0.9"
    elif val <= 1.0:
        return "<1.0"


def calculate_indices(selected_vector, index):
    results, _ = match_query(indices_name=[index.name], query_id=selected_vector)
    knn_indices, scores = list(zip(*results[list(results)[0]]))
    return knn_indices, scores


def get_index(
    embedding_space_name: str, model: str = "lsh", similarity: str = "cosine"
) -> Tuple[np.ndarray, List[Dict[str, str]], ElasticSearchIndexModel]:

    embedding_space = Query(EmbeddingSpaceModel).get(name=embedding_space_name)[0]
    embeddings = load_embeddings_from_numpy(embedding_space.npz_path)
    rows, _, _ = load_csv_with_headers(embedding_space.dataset.csv_path)

    index = Query(ElasticSearchIndexModel).get(
        embedding_space=embedding_space, model=model, similarity=similarity
    )[0]

    return embeddings, rows, index


def compute_similarity(
    embedding_space_name_1: str,
    embedding_space_name_2: str,
    similarity_1: str,
    similarity_2: str,
    df_1: pd.DataFrame,
    df_2: pd.DataFrame,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Compute similarity between two embedding spaces."""

    similarity_1 = "euclidean" if similarity_1 == "l2" else "cosine"
    similarity_2 = "euclidean" if similarity_2 == "l2" else "cosine"

    knn = KNNBulkOperations(embedding_space_name_1).space_similarity(
        embedding_space_name_1,
        similarity_1,
        embedding_space_name_2,
        similarity_2,
    )
    assert knn != {}
    knn_class = [map_coincidence(i) for i in knn.values()]
    df_1["coincidence"] = knn_class
    df_2["coincidence"] = knn_class

    return knn, df_1, df_2


def make_similarity_histogram(knn: dict) -> go.Figure:
    """Make embeddings similarity histogram from knn dict."""
    bar_heights, bin_edges = np.histogram(list(knn.values()), np.arange(0.0, 1.1, 0.1))
    bins = [
        f"{bin_edges[i]:.1f} - {bin_edges[i + 1]:.1f}"
        for i in range(len(bin_edges) - 1)
    ]
    df = pd.DataFrame(
        {
            "Frecuency": bar_heights / len(knn),
            "IoU": bins,
        }
    )

    value_color_mapping = zip(bins, RdYlBu10)
    fig = px.bar(
        df,
        x="IoU",
        y="Frecuency",
        color="IoU",
        color_discrete_map=dict(value_color_mapping),
    )

    return fig
