import csv
import sys
from typing import List, Sequence, Tuple

import numpy as np
from vectory.exceptions import IsNotAColumnError


def load_csv_with_headers(
    csv_path: str,
    id_field: str = "_idx",
) -> Tuple[List[dict], dict, Sequence[str]]:
    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames

        if header is None:
            raise ValueError("No header in the CSV file")

        if id_field not in header and id_field != "_idx":
            raise IsNotAColumnError(
                f"Error: {id_field} is not a columns of the given csv"
            )

        header = list(header)

        try:
            rows = list(reader)
        except csv.Error as e:
            sys.exit(f"file {csv_path}, line {reader.line_num}: {e}")

        if id_field == "_idx":
            header.append(id_field)
            for i, row in enumerate(rows):
                row[id_field] = str(i)

        header_mapping = {keyword: {"type": "keyword"} for keyword in header}
        header_mapping[id_field]["store"] = True  # type: ignore

    return rows, header_mapping, header


def load_embeddings_from_numpy(embeddings_path: str) -> np.ndarray:
    if embeddings_path.endswith(".npy"):
        embeddings = np.load(embeddings_path)
    elif embeddings_path.endswith(".npz"):
        embeddings = np.load(embeddings_path)
        embeddings = embeddings[embeddings.files[0]]
    else:
        raise NotImplementedError("Unknown file extension for embeddings file")
    return embeddings
