"""Shared DIGAT metadata, semantic-graph specifications and configuration."""
import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DIGAT"))
from digat_run import DEFAULT_SETTINGS, verify_upstream
from nnr_artifacts import fingerprint


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def shared_spec(data_root, embedding_root, sag_model, seed):
    model_manifest = json.loads((sag_model / "TORS_ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    return {
        "news": {split: fingerprint(data_root / split / "news.tsv") for split in ("train", "dev", "test")},
        "glove": fingerprint(embedding_root / "glove.6B.300d.txt"),
        "sag_asset": model_manifest,
        "seed": seed,
        "parameters": {key: DEFAULT_SETTINGS[key] for key in
                       ("word_threshold", "max_title_length", "word_embedding_dim", "SAG_hops", "SAG_neighbors")},
        "upstream": verify_upstream(),
        "glove_override": "6B.300d instead of upstream 840B.300d; shared across the six-model diagnostic experiment",
    }


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_for(data_root, seed, batch_size=16, epochs=1):
    values = dict(DEFAULT_SETTINGS)
    values.update(dataset="MIND-small", seed=seed, batch_size=batch_size,
                  epoch=epochs, dropout_rate=0.2, local_rank=-1, device_id=0, mode="train")
    for split in ("train", "dev", "test"):
        values[split + "_root"] = str(data_root / split)
    values["news_graph_size"] = 1 + values["SAG_neighbors"] + values["SAG_neighbors"] * (values["SAG_neighbors"] - 1)
    result = SimpleNamespace(**values)
    result.attribute_dict = dict(values)
    return result




