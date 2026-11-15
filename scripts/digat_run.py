#!/usr/bin/env python
"""DIGAT source verification, configuration and checkpoint utilities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from digat_prepare import sha256_file, validate_prepared

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "DIGAT"
COMMIT = "0408369f20768b34c568f2db3937f276803a2868"
SAG_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Explicitly documented upstream reference defaults, not recovered TORS settings.
DEFAULT_SETTINGS = {
    "news_encoder": "MSA", "graph_encoder": "DIGAT",
    "word_threshold": 3, "max_title_length": 32, "negative_sample_num": 4,
    "max_history_num": 50, "batch_size": 64, "lr": 0.0001,
    "weight_decay": 0.0, "gradient_clip_norm": 1.0,
    "dev_criterion": "avg", "early_stopping_epoch": 5,
    "word_embedding_dim": 300, "cnn_method": "naive", "cnn_kernel_num": 400,
    "cnn_window_size": 3, "MSA_head_num": 16, "MSA_head_dim": 25,
    "attention_dim": 256, "graph_depth": 3, "SAG_hops": 2, "SAG_neighbors": 5,
}


def verify_upstream(directory=UPSTREAM):
    metadata = json.loads((directory / "UPSTREAM.json").read_text(encoding="utf-8"))
    if metadata.get("commit") != COMMIT or metadata.get("source_modified") is not False:
        raise ValueError("Unexpected upstream DIGAT provenance")
    for name, digest in metadata["sha256"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError(f"Vendored upstream DIGAT file changed: {name}")
    return metadata


def effective_settings(args):
    settings = dict(DEFAULT_SETTINGS)
    if args.settings:
        supplied = json.loads(args.settings.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_SETTINGS):
            raise ValueError("Settings JSON may override only documented DEFAULT_SETTINGS keys")
        settings.update(supplied)
    if args.seed < 0 or args.seed >= 2**32 or args.epochs <= 0 or not 0 <= args.dropout < 1:
        raise ValueError("Require 0 <= seed < 2**32, epochs > 0 and 0 <= dropout < 1")
    for name, default in DEFAULT_SETTINGS.items():
        value = settings[name]
        if isinstance(default, int):
            if type(value) is not int or value < (0 if name == "early_stopping_epoch" else 1):
                raise ValueError(f"Invalid integer setting: {name}")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Invalid numeric setting: {name}")
            if not __import__("math").isfinite(value):
                raise ValueError(f"Non-finite numeric setting: {name}")
    if settings["lr"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Require lr > 0 and weight_decay >= 0")
    if settings["news_encoder"] not in ("MSA", "CNN") or settings["graph_encoder"] != "DIGAT":
        raise ValueError("This adapter supports official DIGAT with MSA or CNN, not ablation models")
    if settings["cnn_method"] != "naive":
        raise ValueError("Only the upstream default naive CNN branch is supported by this adapter")
    if settings["word_embedding_dim"] not in (50, 100, 200, 300):
        raise ValueError("Unsupported GloVe dimension")
    if settings["dev_criterion"] not in ("auc", "mrr", "ndcg5", "ndcg10", "avg"):
        raise ValueError("Unsupported dev criterion")
    if settings["SAG_neighbors"] < 2:
        raise ValueError("SAG_neighbors must be at least 2")
    settings.update(seed=args.seed, epoch=args.epochs, dropout_rate=args.dropout,
                    device_id=args.device_id, local_rank=-1, mode=args.mode)
    return settings


def glove_file(glove_dir, dim):
    name = "glove.840B.300d.txt" if dim == 300 else f"glove.6B.{dim}d.txt"
    for filename in (name + ".pt", name):
        candidate = glove_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Supply local {name} or {name}.pt in {glove_dir}; no GloVe download is performed")


def preflight(args):
    if args.mode != "train":
        raise ValueError(
            "Standalone dev/test is disabled until checkpoint and training-time "
            "vocabulary/graph/preprocessing manifests can be verified; use train with its automatic test"
        )
    prepared = args.prepared.resolve()
    manifest = validate_prepared(prepared)
    vendor = verify_upstream()
    settings = effective_settings(args)
    if args.output.exists():
        raise FileExistsError(f"Use a new run output directory; caches are never reused: {args.output}")
    output = args.output.resolve()
    if output == prepared or prepared in output.parents:
        raise ValueError("Run output must be outside the prepared input directory")
    glove = glove_file(args.glove_dir.resolve(), settings["word_embedding_dim"])
    if args.sag_model is not None:
        sag_model = args.sag_model.resolve()
        if not sag_model.is_dir() or not (sag_model / "modules.json").is_file():
            raise FileNotFoundError("Local SAG model must be a saved all-mpnet-base-v2 SentenceTransformer directory")
    elif args.allow_model_download:
        sag_model = SAG_MODEL
    else:
        raise ValueError("Supply --sag-model local directory or explicitly --allow-model-download")
    if args.mode != "train" and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise FileNotFoundError("dev/test require an existing trusted DIGAT --checkpoint")
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("--checkpoint is for dev/test; this adapter does not resume training")
    return manifest, vendor, settings, glove, str(sag_model)




def write_truth(behavior_file, target):
    with behavior_file.open(encoding="utf-8") as stream, target.open("w", encoding="utf-8", newline="\n") as out:
        for index, line in enumerate(stream):
            labels = [int(item.rsplit("-", 1)[1]) for item in line.rstrip("\r\n").split("\t")[4].split(" ")]
            out.write(("" if index == 0 else "\n") + str(index + 1) + " " + json.dumps(labels, separators=(",", ":")))


def make_config_class(base_config, prepared, settings, manifest):
    """Subclass configuration only; preserve official model/corpus/graph/trainer."""
    class TORSConfig(base_config):
        def parse_argument(self):
            attrs = dict(settings)
            attrs.update(
                # This is an internal label-handling compatibility branch. It is
                # NOT a claim that a MIND-200K input has become MIND-Small.
                dataset="MIND-small",
                source_dataset=manifest["dataset"],
                source_dataset_display_name=manifest["dataset_display_name"],
                source_protocols=manifest["protocols"],
                historical_R2_settings_verified=False,
                adapter_profile="new-R3-integration-unverified-against-R2",
                dev_model_path="", test_model_path="", test_output_file="",
            )
            self.attribute_dict = attrs
            for key, value in attrs.items():
                setattr(self, key, value)
            for split in ("train", "dev", "test"):
                setattr(self, split + "_root", str(prepared / "dataset" / split))
            self.news_graph_size = 1
            neighbors = 1
            for depth in range(self.SAG_hops):
                neighbors *= self.SAG_neighbors if depth == 0 else self.SAG_neighbors - 1
                self.news_graph_size += neighbors

        def preliminary_setup(self):
            from MIND_corpus import MIND_Corpus
            name = self.news_encoder + "-" + self.graph_encoder
            for path in (
                f"configs/{self.dataset}/{name}", f"models/{self.dataset}/{name}",
                f"best_model/{self.dataset}/{name}", f"results/{self.dataset}/{name}",
                f"dev/{self.dataset}/ref", f"dev/{self.dataset}/res/{name}",
                f"test/{self.dataset}/ref", f"test/{self.dataset}/res/{name}",
            ):
                Path(path).mkdir(parents=True, exist_ok=True)
            for split in ("dev", "test"):
                write_truth(Path(getattr(self, split + "_root")) / "behaviors.tsv",
                            Path(split) / self.dataset / "ref" / "truth.txt")
            # No upstream dataset download/preparation call.
            MIND_Corpus.preprocess(self)

    return TORSConfig


def stage_checkpoint(checkpoint, output, expected_sha256, model_name):
    # Upstream main derives a result directory from this path. Keep it relative
    # so a Windows drive colon cannot become an invalid output-directory name.
    relative = Path("input_checkpoint") / model_name
    destination = output / relative
    destination.parent.mkdir()
    shutil.copyfile(checkpoint, destination)
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError("Checkpoint changed while copying")
    return relative.as_posix()


def package_versions():
    result = {}
    for name in ("torch", "torchtext", "torch-scatter", "sentence-transformers", "transformers",
                 "tokenizers", "numpy", "scikit-learn", "nltk"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def execute(args):
    manifest, vendor, settings, glove, sag_model = preflight(args)
    summary = {
        "status": "input-check-only" if args.check_only else "starting",
        "dataset": manifest["dataset_display_name"], "protocols": manifest["protocols"],
        "upstream_internal_dataset_key": "MIND-small (labeled-test compatibility only)",
        "effective_settings": settings, "historical_R2_identity": "unverified",
        "validation_policy": "training_matched",
        "matched_candidate_profile": manifest.get("matched_candidate_profile"),
        "gpu_execution_verified": False, "package_versions": package_versions(),
    }
    if args.check_only:
        print(json.dumps(summary, indent=2))
        return
    if not args.acknowledge_unverified_profile:
        raise ValueError("Use --acknowledge-unverified-profile; this is not a recovered historical R2 DIGAT implementation")
    # Fail before copying multi-GB assets if the execution environment is absent.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Official DIGAT requires a CUDA GPU; input-check-only is not a GPU execution test")
    output, prepared = args.output.resolve(), args.prepared.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(summary, created_utc=datetime.now(timezone.utc).isoformat(),
                   prepared_manifest_sha256=sha256_file(prepared / "manifest.json"),
                   source_manifest=manifest, upstream_commit=vendor["commit"],
                   adapter_source_sha256={name: sha256_file(REPO_ROOT / "scripts" / name)
                                          for name in ("digat_prepare.py", "digat_run.py", "training_validation.py", "prepare_development_candidates.py")},
                   python=sys.version, python_hash_seed=os.environ.get("PYTHONHASHSEED"),
                   cuda_device=torch.cuda.get_device_name(args.device_id),
                   glove_source=str(glove), glove_sha256=sha256_file(glove),
                   sag_model=sag_model, allow_model_download=args.allow_model_download,
                   checkpoint_sha256=sha256_file(checkpoint) if checkpoint else None)
    if args.sag_model:
        runtime["sag_local_files_sha256"] = {
            p.relative_to(args.sag_model.resolve()).as_posix(): sha256_file(p)
            for p in sorted(args.sag_model.resolve().rglob("*")) if p.is_file()
        }
    runtime_path = output / "run_manifest.json"

    def save_manifest():
        runtime_path.write_text(json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    save_manifest()
    previous_cwd = Path.cwd()
    try:
        # Isolate generated pretrained-vector cache as well as DIGAT caches.
        glove_cache = output / "glove"
        glove_cache.mkdir()
        shutil.copyfile(glove, glove_cache / glove.name)
        if sha256_file(glove_cache / glove.name) != runtime["glove_sha256"]:
            raise RuntimeError("GloVe source changed while copying")
        os.environ["HF_HOME"] = str(output / "hf-cache")
        os.environ["HF_HUB_OFFLINE"] = "0" if args.allow_model_download else "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "0" if args.allow_model_download else "1"
        os.chdir(output)
        sys.path.insert(0, str(UPSTREAM))
        import config as upstream_config
        import MIND_corpus
        import construct_SAG
        original_glove = MIND_corpus.GloVe

        def local_glove(*call_args, **kwargs):
            kwargs["cache"] = str(glove_cache)
            return original_glove(*call_args, **kwargs)

        MIND_corpus.GloVe = local_glove
        construct_SAG.pretrain_model_name = sag_model
        local_checkpoint = None
        if checkpoint is not None:
            local_checkpoint = stage_checkpoint(
                checkpoint, output, runtime["checkpoint_sha256"], settings["news_encoder"] + "-DIGAT"
            )
        TORSConfig = make_config_class(upstream_config.Config, prepared, settings, manifest)
        config = TORSConfig()
        import main as upstream_main
        corpus = MIND_corpus.MIND_Corpus(config)
        if args.mode == "train":
            trainer = upstream_main.train(config, corpus)
            config.test_model_path = f"best_model/{config.dataset}/{trainer.model.model_name}/#{trainer.run_index}/{trainer.model.model_name}"
            config.test_output_file = f"results/{config.dataset}/{trainer.model.model_name}/#{trainer.run_index}-test"
            upstream_main.test(config, corpus)
        elif args.mode == "dev":
            config.dev_model_path = local_checkpoint
            upstream_main.dev(config, corpus)
        else:
            config.test_model_path = local_checkpoint
            config.test_output_file = f"results/{config.dataset}/{config.news_encoder}-DIGAT/evaluation-test"
            config.run_index = 0
            upstream_main.test(config, corpus)
        runtime["status"] = "completed"
        runtime["gpu_execution_verified"] = True
        runtime["historical_R2_identity"] = "unverified even after this run"
    except BaseException as error:
        runtime["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        runtime["error_type"] = type(error).__name__
        runtime["error"] = str(error)
        raise
    finally:
        os.chdir(previous_cwd)
        runtime["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest()


