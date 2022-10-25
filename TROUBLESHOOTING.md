# Troubleshooting

## Problems starting Elasticsearch

Most issues in Vectory with Elasticsearch are related to Docker. Vectory assumes you have Docker [installed](https://docs.docker.com/get-docker/) with Docker Compose version  `>=2.3.4` and that the Docker daemon is running.

### Using an old version of Docker Compose

You can check your version of Docker Compose by running:
```
docker compose version
```

You can follow [Docker's official guide](https://docs.docker.com/compose/install) to update Docker Compose or install it altogether.

## Problems with the database

### Locked database

You can't have more than one instance of Vectory running at the same time. If you try to start Vectory while it's already running, you'll get an error message like this:
```bash
sqlite3.OperationalError: database is locked
```
Just stop the running instance of Vectory and try again.


### Index already exists

If adding an index to the database fails in a way we haven't anticipated, it might happen that the index is loaded in Elasticsearch but not in the database. This can be checked by comparing the indices on both places:

```python
from vectory.db.models import ElasticSearchIndexModel, Query
from vectory.es.client import ElasticKNNClient

es = ElasticKNNClient()
es_indices = es.list_indices()

db_indices = Query(ElasticSearchIndexModel).get()
```

`es_indices` and `db_indices` should be the same. If they aren't, you can delete the index from Elasticsearch and add it again to the database.

```python
from vectory.es.client import ElasticKNNClient

es = ElasticKNNClient()
es.delete_index(INDEX_NAME)
```

## Import errors

### No module `pkg_resources`

The Python module `setuptools` has errors if installed with `pip` or `poetry`. You can fix this with:

```bash
wget https://bootstrap.pypa.io/ez_setup.py -O - | python
```