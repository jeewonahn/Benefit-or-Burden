"""Synthetic CPU tests; no MIND records or model downloads are used."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from context_embeddings import prepare_context_embeddings


class ContextEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs, self.outputs = {}, {}
        for split in ("train", "dev", "test"):
            folder = self.root / split
            folder.mkdir()
            source = folder / "entity_embedding.vec"
            source.write_text("Q1\t1\t3\nQ2\t3\t7\nQ3\t9\t11\n", encoding="utf-8")
            self.inputs[split] = source
            self.outputs[split] = folder / "context_embedding.vec"
        self.graph = self.root / "wikidata-graph.tsv"
        self.graph.write_text("Q1\tr\tQ2\nQ1\tr2\tQ2\nQ2\tr\tQ1\nQ3\tr\tUNKNOWN\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_builder(self, graph=None):
        return prepare_context_embeddings(self.inputs, graph or self.graph, self.outputs, dimension=2)

    def read_vectors(self):
        return {parts[0]: tuple(map(float, parts[1:]))
                for parts in (row.split() for row in self.outputs["train"].read_text().splitlines())}

    def test_neighbor_mean_and_provenance(self):
        original = self.inputs["train"].read_bytes()
        receipt = self.run_builder()
        self.assertEqual(self.read_vectors(), {"Q1": (2, 5), "Q2": (2, 5), "Q3": (9, 11)})
        self.assertEqual(receipt["graph_lines"], 4)
        self.assertEqual(receipt["directed_unique_neighbors"], 2)
        self.assertEqual(self.inputs["train"].read_bytes(), original)
        self.assertEqual(receipt["outputs"]["train"]["entities"], 3)
        json.dumps(receipt)

    def test_self_loop_matches_upstream(self):
        self.graph.write_text("Q1\tr\tQ1\nQ1\tr\tQ2\n", encoding="utf-8")
        self.run_builder()
        self.assertAlmostEqual(self.read_vectors()["Q1"][0], 5 / 3)
        self.assertAlmostEqual(self.read_vectors()["Q1"][1], 13 / 3)

    def test_valid_zip(self):
        archive = self.root / "graph.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("graph/wikidata-graph.tsv", self.graph.read_bytes())
        receipt = self.run_builder(archive)
        self.assertEqual(receipt["graph_member"]["name"], "graph/wikidata-graph.tsv")
        self.assertEqual(self.read_vectors()["Q1"], (2, 5))

    def test_unsafe_zip_rejected_without_outputs(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("../wikidata-graph.tsv", "Q1\tr\tQ2\n")
        with self.assertRaises(ValueError):
            self.run_builder(archive)
        self.assertFalse(any(path.exists() for path in self.outputs.values()))

    def test_cross_split_conflict_rejected(self):
        self.inputs["test"].write_text("Q1\t2\t3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.run_builder()

    def test_duplicate_within_split_rejected(self):
        self.inputs["train"].write_text("Q1\t1\t3\nQ1\t1\t3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_builder()

    def test_nonfinite_rejected(self):
        self.inputs["dev"].write_text("Q1\tnan\t3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.run_builder()

    def test_malformed_graph_rejected(self):
        self.graph.write_text("Q1\tQ2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected subject"):
            self.run_builder()

    def test_no_overwrite(self):
        self.outputs["dev"].write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_builder()
        self.assertEqual(self.outputs["dev"].read_text(), "keep")
        self.assertFalse(self.outputs["train"].exists())

    def test_stable_output(self):
        self.run_builder()
        first = {split: path.read_bytes() for split, path in self.outputs.items()}
        for path in self.outputs.values():
            path.unlink()
        self.run_builder()
        self.assertEqual(first, {split: path.read_bytes() for split, path in self.outputs.items()})


if __name__ == "__main__":
    unittest.main()
