<p align="center">
  <img src="https://pento.ai/images/vectory-banner.png" alt="Vectory">
</p>

<p align="center">
    <b> An embedding evaluation toolkit </b>
</p>

<p align="center">
    <a href="https://pypi.org/project/vectory" target="_blank">
        <img src="https://img.shields.io/pypi/v/vectory?color=%2334D058&label=pypi%20package" alt="Package version">
    </a>
    <a href="https://pypi.org/project/vectory" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/vectory.svg?color=%2334D058" alt="Supported Python versions">
    </a>
</p>

Vectory provides a collection of tools to **track and compare embedding versions**.

Being able to visualize and register each experiment is a crucial part of developing successful models. Vectory is a tool designed by and for machine learning engineers to handle embedding experiments with little overhead.

### Key features:
- **Embedding linage**. Keep track of what data and models were used to generate embeddings.
- **Compare performance**. Compare metrics between different vector spaces.
- **Ease of use**. Easy usage through the CLI, Python and GUI interfaces.
- **Extensibility**. It was built with extensibility in mind.
- **Persistence**. Simple local state persistence using SQLite.

# Table of Contents
1. [Installation](#installation)
2. [Demo](#demo)
3. [Usage](#usage)
4. [Troubleshooting](troubleshooting.md)
5. [License](#license)


# Installation

All you need for Vectory to run is to install the package and Elasticsearch. You can install the package using pip:

```console
pip install vectory
```

## Set up Elasticsearch

What is Elasticsearch? It's a free high performance search engine, which is used for any kind of data.

Vectory uses Elasticsearch to load embeddings and then search for them.

To start the engine you will need to install Docker and start its daemon.
After that, just run:
```console
vectory elastic up --detach
```

and you can turn it off with:
```console
vectory elastic down
```

# Demo

After installing vectory with the GUI dependencies, you can play with the demo cases to get a feel of the toolkit.
- Tiny-imagenet computer vision dataset embeddings made from pretrained models ResNet50 and ConvNext-tiny.
- Imdb nlp dataset embeddings made from pretrained models BERT and RoBERTa.

In order to download the data and set up the demo, run the following command:
```console
vectory demo 
```
You can specify the demo dataset with the `--dataset-name` argument.

Run the Streamlit viualization app:

```console
vectory run
```

# Usage

The key concepts needed to use Vectory are **datasets**, **experiments** and **embedding spaces**.

A **dataset** is just a collection of data. You could have evaluation or training datasets. Evaluation datasets are required for Vectory to run, whereas training datasets are optional, desired for tracking purposes.

Datasets are defined with a csv file. The csv file should have a header row, followed by a row for each data point. The columns may contain any information about the data point, but it is recommended that the first column is an identifier for the data point. The next columns could be labels, features, or any other information.

An **experiment** is a machine learning model which has been trained with a particular dataset. You could create different experiments by varying the model and the dataset. As well as the training datasets, the experiments are optional and desired for tracking purposes.

Together, they form an **embedding space**, which is just a 2-dimensional array with all the generated vectors (or features or embeddings) for a particular dataset using a particular experiment. They can be either `.npz` files or `.npy` files, we'll refer to them as `.npz` for simplicity. It must follow the same order as the evaluation dataset csv file.

<details markdown="1">
<summary> <b> Example </b> </summary>

You could have an experiment, such as a ResNet model trained with the dataset Data1. Let’s call the generated embedding space ES1. But either you split your data or you get new data once in a while (or both), so this experiment will not only be used in a static dataset. You might want to use this experiment on Data2 then, generating a particular embedding space called ES2.

Vectory helps you to organize and analyze the obtained embeddings for each dataset and experiment.

</details>

---

## Command Line Interface

### Create

Create datasets, experiments and embedding spaces:
```console
vectory add --dataset [path_to_csv] --embeddings [path_to_npz]
```

This is the most simple way to add them. In case you want to track your tests, you can specify the names of the elements, the dimension of the embedding space and the parameters of the model. You can see all the options with the `--help` flag.

### Load

Embedding spaces are mapped to Elasticsearch **indices**. To load the embeddings to Elasticsearch when creating the embedding space with the previous command, add `--load ` after designating the dataset, the embedding space and the parameters. This option for the `add` command only works for the default loading options. If you want to load the embeddings with different options, you can use the `load` command.

Load independentely an embedding space to Elasticsearch:

```console
vectory embeddings load [index_name] [embedding_space_name]
```

You can specify the model name, the similarity function, the number of threads, the chunk size and the hyperparameters for the kNN search. You can see all the options with the `--help` flag.


### Search

Get all your datasets, experiments, embedding spaces and indices:
```console
vectory ls
```
List all the indices:
```console
vectory embeddings list-indices
```

### Delete

Delete datasets:
```console
vectory dataset delete [dataset_name]
```

Experiments:
```console
vectory experiment delete [experiment_name] 
```

Embedding Spaces:
```console
vectory space delete [embedding_space_name] 
```

You can delete elements associated to these objects and their respective indices adding `--recursive`.

Indices:
```console
vectory embeddings delete-index [index_name]
```

All indices:
```console
vectory embeddings delete-all-indices
```

### Comparing embedding spaces

With Vectory you can measure how similar two embedding spaces are. The similarity between two embedding spaces is the mean of the local neighbourhood similarity of every point, which is the IoU of the 10 nearest neighbours.

Basically, in order to compare 2 embedding spaces Vectory computes the 10 nearest neighbours for every data point for both embedding spaces, get the IoU for each group of 10 nearest neighbours obtained and shows the distribution of the IoU values. Also, we compute the mean of the IoU values in order to provide a single value to compare the two embedding spaces.

More info about comparing embedding spaces [here](http://vis.csail.mit.edu/pubs/embedding-comparator/).

Compare two embedding spaces using:

```console
vectory compare [embedding_space_1_name] [embedding_space_2_name] --precompute
```

You can specify the metric to use for kNN search in each of the embedding spaces, calculate similarity histogram and allow precoumpute.


## Python API

### Create

Create datasets, experiments and an embedding space from them.

```python
from vectory.datasets import Dataset
from vectory.experiments import Experiment
from vectory.spaces import EmbeddingSpace

dataset = Dataset.get_or_create(csv_path=CSV_PATH, name=DATASET_NAME)

train_dataset = Dataset.get_or_create(csv_path=TRAIN_CSV_PATH, name=TRAIN_DATASET_NAME)

experiment = Experiment.get_or_create(
    train_dataset=TRAIN_DATASET_NAME,
    model=MODEL_NAME,
    name=EXPERIMENT_NAME,
)

embedding_space = EmbeddingSpace.get_or_create(
    npz_path=NPZ_PATH,
    dims=EMBEDDINGS_DIMENSIONS,
    experiment=EXPERIMENT_NAME,
    dataset=DATASET_NAME,
    name=EMBEDDING_SPACE_NAME,
)
```

The train dataset is optional, but it is recommended to track the training process.

Load an index on elastic search for an embedding space:

```python
from vectory.indices import load_index

load_index(
    index_name=INDEX_NAME,
    embedding_space_name=EMBEDDING_SPACE_NAME,
)
```
The `dataset`, `experiment` and `embedding_space` objects have the `.model.name` attribute, so both the variable and the attribute can be used for specifying the name.

Additionally, you can specify the desired mapping to load the index with. This determies whether `cosine` or `euclidean` similarity will be used for the kNN search, as well as the model for the kNN search. Using an `exact` model instead of the `lsh` option will make the search slower, but more accurate. The `lsh` model and the `cosine` similarity are the default options. To see all the available mappings, check the possible options from `vectory.es.api.Mapping`.

### Search

Get all your datasets, experiments, embedding spaces and indices:
```python
from vectory.db.models import (
    DatasetModel,
    ElasticSearchIndexModel,
    EmbeddingSpaceModel,
    ExperimentModel,
    Query,
)

datasets = Query(DatasetModel).get()
experiments = Query(ExperimentModel).get()
spaces = Query(EmbeddingSpaceModel).get()
indices = Query(ElasticSearchIndexModel).get()
```

You can also get a specific dataset, expeiment, space or index by specifying an attribute:
```python
dataset = Query(DatasetModel).get(name=DATASET_NAME)[0]
```

### Delete

Delete old datasets and its indices if wanted:

```python
from vectory.db.models import  DatasetModel, Query

dataset = Query(DatasetModel).get(name=DATASET_NAME)[0]
dataset.delete_instance(recursive=True)
```

Keep in mind that if the `recursive` option is set to `True`, the experiments, spaces and indices associated with the dataset will be deleted as well.

The same can be done for experiments, embedding spaces and indices by using the `delete_instance` method on the correct object.

### Compare

With Vectory you can measure how similar two embedding spaces are. The similarity between two embedding spaces is the mean of the local neighbourhood similarity of every point, which is the IoU of the 10 nearest neighbours.  More info about comparing embedding spaces [here](http://vis.csail.mit.edu/pubs/embedding-comparator/).

Compare two embedding spaces:

```python
from vectory.spaces import compare_embedding_spaces

similarity, _, fig, _ = compare_embedding_spaces(
    embedding_space_a=EMBEDDING_SPACE_NAME_1,
    embedding_space_b=EMBEDDING_SPACE_NAME_2,
    metric_a=METRIC_A,
    metric_b=METRIC_B,
    allow_precompute_knn=True,
)
```

The `metric_a` and `metric_b` parameters are either `euclidean` or `cosine`. The `allow_precompute_knn` parameter is set to `True` to allow precomputing the bulk operations for the similarity computation.

The `spaces_similarity` variable contains the similarity between the two embedding spaces. The `id_similarity_dict` variable contains the similarity scores for every point in the embedding spaces.

An additional argument can be passed to the `compare_embedding_spaces` function, which is `histogram`. If set to `True`, the function will show a histogram of the similarity scores, otherwise, an empty figure is returned. The `fig` and `ax` variables are the figure and axis of the histogram.

### Reduce dimensionality

Reduce the dimensionality to 2D of an embedding space:

```python
from vectory.visualization.utils import calculate_points, get_index

# Get the embedding space data
embeddings, rows, index = get_index(
    EMBEDDING_SPACE_NAME, model=MODEL, similarity=SIMILARITY_METHOD
)

# Reduce the dimensionality
df = calculate_points(DIMENSIONAL_REDUCTION_MODEL, embeddings, rows)
```

The `calculate_points` function reduces the dimensionality of the embeddings using the `DIMENSIONAL_REDUCTION_MODEL` model. It can be either `UMAP`, `PCA` or `PCA + UMAP`. It returns a DataFrame with the reduced dimensionality points and the data contained in the dataset's csv.

### Get similar indices

Get the most similar indices for a given embedding:

```python
from vectory.indices import match_query

# Get the most similar indices for a sample embedding
similarity_results, _ = match_query(indices_name=[INDEX_NAME], query_id=EMBEDDING_INDEX)
```

The `match_query` function returns the most similar indices for a given embedding and the index of the embedding. The `indices_name` parameter is a list of indices names, and the `query_id` parameter is the id of the embedding to search for. From these results, you can get the most similar indices and their scores. The `similarity_results` variable contains a dictionary with the indices names as keys and a list of tuples with the most similar indices and their scores as values.

## Visualization

Once you have loaded your datasets, experiments and empedding spaces, you can analyze the results either by visualizing them on our Streamlit app or by following the Python API documentation and getting the indices.

### Streamlit
Visualize your embedding spaces on a local Streamlit app with:
```console
vectory run
```
The GUI dependencies are required to view the Streamlit app.

# License

This project is licensed under the terms of the MIT license.
