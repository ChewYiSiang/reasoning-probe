"""Run the MuCoCo tester without a live MongoDB connection.

Colab cannot always reach Atlas: the TLS handshake fails behind some network paths, and
the address changes every session. The tasks themselves are small and unchanging, so this
exports them once to a file and serves them back through the three calls the tester makes
on a collection: count_documents, find_one and find.

    # on your laptop, with Mongo reachable
    python -m probe.offline_db --export tasks_cruxeval.json

    # in Colab, before building the tester
    from probe.offline_db import OfflineCollection, attach
    attach(llmtester, "/content/tasks_cruxeval.json")
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterator


class OfflineCollection:
    """The small part of a MongoDB collection that the tester actually uses."""

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._by_id = {document["_id"]: document for document in documents}

    def count_documents(self, query: dict | None = None) -> int:
        return len(self.documents)

    def find_one(self, query: dict) -> dict | None:
        if "_id" in query:
            return self._by_id.get(query["_id"])
        return self.documents[0] if self.documents else None

    def find(self, query: dict | None = None, projection: dict | None = None) -> Iterator[dict]:
        """Only the projections the tester asks for are honoured, which is enough."""
        for document in self.documents:
            if not projection:
                yield document
                continue
            wanted = {key for key, keep in projection.items() if keep and key != "_id"}
            result: dict[str, Any] = {key: document[key] for key in wanted if key in document}
            if projection.get("_id", 1):
                result["_id"] = document["_id"]
            yield result


def export(collection, path: str) -> int:
    """Write every document of a live collection to a JSON file."""
    documents = list(collection.find({}))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(documents, handle)
    return len(documents)


def load(path: str) -> OfflineCollection:
    with open(path, encoding="utf-8") as handle:
        return OfflineCollection(json.load(handle))


def make_tester(tester_class, path: str):
    """Build a tester that never touches MongoDB.

    Their constructor connects to Mongo and raises if it cannot, but the only thing it
    sets up is `question_database`. Creating the object without running the constructor
    and pointing it at the exported file gives a fully working tester: everything else it
    needs is inherited behaviour, not state.
    """
    tester = tester_class.__new__(tester_class)
    tester.question_database = load(path)
    print(f"tester built without MongoDB, {tester.question_database.count_documents()} tasks")
    return tester


def attach(tester, path: str) -> OfflineCollection:
    """Point a tester at the exported file instead of the database."""
    collection = load(path)
    tester.question_database = collection
    print(f"tester now reading {collection.count_documents()} tasks from {path}")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, help="file to write")
    parser.add_argument("--database", default=None, help="defaults to MONGODB_BENCHMARK_DATABASE")
    parser.add_argument("--collection", default=None, help="defaults to MONGODB_CRUXEVAL_COLLECTION")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    client = MongoClient(os.environ["MONGODB_URI"])
    database = client[args.database or os.environ.get("MONGODB_BENCHMARK_DATABASE",
                                                     "Base_Questions_DB")]
    collection = database[args.collection or os.environ.get("MONGODB_CRUXEVAL_COLLECTION",
                                                            "CruxEval_Input_Output")]
    print(f"exported {export(collection, args.export)} tasks to {args.export}")


if __name__ == "__main__":
    main()
