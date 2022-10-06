"""Forked from
https://github.com/alexklibisz/elastiknn/tree/master/client-python/elastiknn"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import List


class Similarity(Enum):
    L2 = 1
    Cosine = 2


class Vec:
    @dataclass(init=False, frozen=True)
    class Base(ABC):
        pass

    @dataclass(frozen=True)
    class DenseFloat(Base):
        values: List[float]

        def to_dict(self):
            return self.__dict__

        def __len__(self):
            return len(self.values)

    @dataclass(frozen=True)
    class Indexed(Base):
        index: str
        id: str
        field: str = "embedding"

        def to_dict(self):
            return self.__dict__


class Mapping:
    @dataclass(init=False, frozen=True)
    class Base(ABC):
        dims: int

    @dataclass(frozen=True)
    class DenseFloat(Base):
        dims: int

        def to_dict(self):
            return {
                "type": "elastiknn_dense_float_vector",
                "elastiknn": {"dims": self.dims},
            }

    @dataclass(frozen=True)
    class CosineLsh(Base):
        dims: int
        L: int
        k: int

        def to_dict(self):
            return {
                "type": "elastiknn_dense_float_vector",
                "elastiknn": {
                    "model": "lsh",
                    "similarity": "cosine",
                    "dims": self.dims,
                    "L": self.L,
                    "k": self.k,
                },
            }

    @dataclass(frozen=True)
    class L2Lsh(Base):
        dims: int
        L: int
        k: int
        w: int

        def to_dict(self):
            return {
                "type": "elastiknn_dense_float_vector",
                "elastiknn": {
                    "model": "lsh",
                    "similarity": "l2",
                    "dims": self.dims,
                    "L": self.L,
                    "k": self.k,
                    "w": self.w,
                },
            }


class NearestNeighborsQuery:
    @dataclass(frozen=True, init=False)
    class Base(ABC):
        vec: Vec.Base
        similarity: Similarity
        field: str

        def with_vec(self, vec: Vec.Base):
            raise NotImplementedError

    @dataclass(frozen=True)
    class Exact(Base):
        vec: Vec.Base
        similarity: Similarity
        field: str = "embedding"

        def to_dict(self):
            return {
                "field": self.field,
                "model": "exact",
                "similarity": self.similarity.name.lower(),
                "vec": self.vec.to_dict(),
            }

        def with_vec(self, vec: Vec.Base):
            return NearestNeighborsQuery.Exact(
                field=self.field, vec=vec, similarity=self.similarity
            )

    @dataclass(frozen=True)
    class CosineLsh(Base):
        vec: Vec.Base
        similarity: Similarity = Similarity.Cosine
        candidates: int = 10
        field: str = "embedding"

        def to_dict(self):
            return {
                "field": self.field,
                "model": "lsh",
                "similarity": self.similarity.name.lower(),
                "candidates": self.candidates,
                "vec": self.vec.to_dict(),
            }

        def with_vec(self, vec: Vec.Base):
            return NearestNeighborsQuery.CosineLsh(
                field=self.field,
                vec=vec,
                similarity=self.similarity,
                candidates=self.candidates,
            )

    @dataclass(frozen=True)
    class L2Lsh(Base):
        vec: Vec.Base
        probes: int = 0
        similarity: Similarity = Similarity.L2
        candidates: int = 10
        field: str = "embedding"

        def to_dict(self):
            return {
                "field": self.field,
                "model": "lsh",
                "similarity": self.similarity.name.lower(),
                "probes": self.probes,
                "candidates": self.candidates,
                "vec": self.vec.to_dict(),
            }

        def with_vec(self, vec: Vec.Base):
            return NearestNeighborsQuery.L2Lsh(
                field=self.field,
                vec=vec,
                probes=self.probes,
                similarity=self.similarity,
                candidates=self.candidates,
            )
