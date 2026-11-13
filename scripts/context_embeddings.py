"""Build NNR entity-context vectors from official MIND entities and Wikidata edges.

The method follows Veason-silverbullet/NNR's prepare_MIND_dataset.py:
an entity's context is the mean of its own vector and each distinct embedded
undirected graph neighbor. No text model, training, or network is required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

METHOD = "nnr_undirected_embedded_neighbors_plus_self_mean_v1"
UPSTREAM = "https://github.com/Veason-silverbullet/NNR/blob/master/prepare_MIND_dataset.py"


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _unchanged(path, before):
    now = Path(path).stat()
    if (now.st_size, now.st_mtime_ns) != before:
        raise ValueError("Input changed while being read: " + str(path))


def _load_embeddings(paths, dimension):
    vectors, split_ids, receipts = {}, {}, {}
    for split, raw_path in paths.items():
        path = Path(raw_path).resolve()
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        ids, seen, digest = [], set(), hashlib.sha256()
        with path.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                digest.update(raw)
                parts = raw.decode("utf-8-sig" if number == 1 else "utf-8").strip().split()
                if len(parts) != dimension + 1:
                    raise ValueError(f"{path}:{number}: expected ID and {dimension} values")
                entity = parts[0]
                if entity in seen:
                    raise ValueError(f"{path}:{number}: duplicate entity {entity}")
                values = tuple(float(value) for value in parts[1:])
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{path}:{number}: non-finite vector")
                if entity in vectors and vectors[entity] != values:
                    raise ValueError("Conflicting cross-split entity vector: " + entity)
                seen.add(entity)
                ids.append(entity)
                vectors[entity] = values
        _unchanged(path, before)
        if not ids:
            raise ValueError("Empty entity embedding file: " + str(path))
        split_ids[split] = ids
        receipts[split] = {"path": str(path), "sha256": digest.hexdigest(),
                           "bytes": before[0], "entities": len(ids)}
    return vectors, split_ids, receipts


@contextmanager
def _graph_stream(path):
    path = Path(path)
    if not zipfile.is_zipfile(path):
        with path.open("rb") as stream:
            yield stream, None
        return
    with zipfile.ZipFile(path) as archive:
        members, names = [], set()
        for member in archive.infolist():
            name = member.filename
            normalized = name.replace("\\", "/")
            pure = PurePosixPath(normalized)
            if (pure.is_absolute() or ".." in pure.parts or ":" in normalized
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe graph archive member: " + name)
            if normalized.casefold() in names:
                raise ValueError("Duplicate graph archive member: " + name)
            names.add(normalized.casefold())
            if not member.is_dir() and pure.name == "wikidata-graph.tsv":
                members.append(member)
        if len(members) != 1:
            raise ValueError("Graph archive must contain exactly one wikidata-graph.tsv")
        member = members[0]
        if member.flag_bits & 1:
            raise ValueError("Encrypted graph archive is unsupported")
        with archive.open(member) as stream:
            yield stream, {"name": member.filename, "bytes": member.file_size,
                           "crc32": format(member.CRC, "08x")}


def prepare_context_embeddings(embedding_files, graph_path, output_paths, dimension=100):
    """Write new context files and return a provenance receipt.

    Caller-owned atomic staging directories are supported. Existing outputs are
    never overwritten. Unknown graph entities are ignored, as in upstream NNR.
    Sorted neighbors make floating-point summation deterministic; this need not
    be byte-identical to the upstream set-iteration implementation.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not embedding_files or set(embedding_files) != set(output_paths):
        raise ValueError("Input and output split names must match and be nonempty")
    inputs = {name: Path(path).resolve() for name, path in embedding_files.items()}
    graph = Path(graph_path).resolve()
    outputs = {name: Path(path).resolve() for name, path in output_paths.items()}
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("Each split needs a distinct output path")
    for output in outputs.values():
        if output.exists() or output in set(inputs.values()) | {graph}:
            raise FileExistsError("Refusing to replace an input or existing output: " + str(output))
        if not output.parent.is_dir():
            raise ValueError("Output parent must exist: " + str(output.parent))
    vectors, split_ids, input_receipts = _load_embeddings(inputs, dimension)
    graph_stat = graph.stat()
    graph_before = (graph_stat.st_size, graph_stat.st_mtime_ns)
    graph_receipt = fingerprint(graph)
    neighbors = {entity: set() for entity in vectors}
    line_count, embedded_edge_lines, digest = 0, 0, hashlib.sha256()
    with _graph_stream(graph) as (stream, member):
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            fields = raw.decode("utf-8-sig" if number == 1 else "utf-8").rstrip("\r\n").split("\t")
            if len(fields) != 3 or not all(fields):
                raise ValueError(f"{graph}:{number}: expected subject, relation, object")
            head, _, tail = fields
            line_count += 1
            if head in vectors and tail in vectors:
                neighbors[head].add(tail)
                neighbors[tail].add(head)
                embedded_edge_lines += 1
    _unchanged(graph, graph_before)
    contexts = {}
    for entity, vector in vectors.items():
        related = sorted(neighbors[entity])
        # A self-loop intentionally contributes the entity again, matching NNR.
        contexts[entity] = tuple(math.fsum([vector[column]] +
            [vectors[neighbor][column] for neighbor in related]) / (1 + len(related))
            for column in range(dimension))
    staged, installed = {}, []
    try:
        for split, output in outputs.items():
            handle, temp_name = tempfile.mkstemp(prefix=".context-", dir=str(output.parent))
            staged[split] = Path(temp_name)
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                for entity in split_ids[split]:
                    stream.write(entity + "\t" + "\t".join(format(value, ".17g")
                                 for value in contexts[entity]) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        for split, output in outputs.items():
            # Exclusive creation also closes the race between preflight and write.
            with output.open("xb") as destination, staged[split].open("rb") as source:
                installed.append(output)
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    destination.write(block)
    except BaseException:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
    return {"method": METHOD, "upstream": UPSTREAM, "dimension": dimension,
            "neighbor_order": "sorted IDs with math.fsum; numerical, not byte identity",
            "duplicate_entity_policy": "identical across splits only; reject within split",
            "graph_filter": "retain undirected unique neighbors with entity embeddings",
            "self_loop_policy": "own vector contributes twice when a self edge exists",
            "inputs": input_receipts, "graph": graph_receipt,
            "graph_member": member, "graph_content_sha256": digest.hexdigest(),
            "graph_lines": line_count, "embedded_edge_lines": embedded_edge_lines,
            "union_entities": len(vectors),
            "directed_unique_neighbors": sum(len(values) for values in neighbors.values()),
            "outputs": {split: dict(fingerprint(path), entities=len(split_ids[split]))
                        for split, path in outputs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Prepared train/dev/test directory containing entity_embedding.vec")
    parser.add_argument("--wikidata-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--dimension", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        (args.output / split).mkdir()
    receipt = prepare_context_embeddings(
        {split: args.data_root / split / "entity_embedding.vec" for split in ("train", "dev", "test")},
        args.wikidata_graph,
        {split: args.output / split / "context_embedding.vec" for split in ("train", "dev", "test")},
        args.dimension)
    (args.output / "context_preparation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": METHOD, "output": str(args.output.resolve()),
                      "entities": receipt["union_entities"]}))


if __name__ == "__main__":
    main()
