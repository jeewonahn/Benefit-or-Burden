# Installation and usage

The workflow is **download -> preprocess -> train -> evaluate -> summarize**. Data, pretrained assets, checkpoints, and generated outputs belong in an external work directory.

## 1. Install the environment

Use Python 3.9. The package profile targets PyTorch 1.12.1/CUDA 11.3, torchtext 0.13.1, torch-scatter 2.0.9, NumPy 1.23.5, sentence-transformers 2.2.2, and transformers 4.30.2.

~~~cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
set "TORS_PYTHON=%CD%\.venv\Scripts\python.exe"
~~~

Binary wheels must match Python, PyTorch, CUDA, and the operating system. The pipeline reports dependencies without installing/upgrading them. DIGAT semantic graph construction requires CUDA. The default MIND tokenizer uses the supplied regex tokenizer.

On Linux, activate with `source .venv/bin/activate` and replace `RUN_MIND_PIPELINE.cmd` with `python -B -u scripts/run_mind_pipeline.py`. Run commands from the repository root.

## 2. Download and preprocess

Review the [MIND terms](https://msnews.github.io/).

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --models all6 --download --accept-mind-license --prepare-only
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --check-only
~~~

Preparation downloads original ZIPs, registers GloVe 6B 300d and pinned MPNet files, and writes masters, proxy, and candidates. It does not train models or construct DIGAT's semantic graph.

| Role | Original dates | Source |
| --- | --- | --- |
| Train | November 9-13 | Official small train archive |
| Development | November 14 | Official small train archive |
| Final test | November 15 | Labeled official small dev archive |

User/time/history request groups remain intact. Dates are checked against the archives. This five-day/one-day/one-day split differs from the manuscript's six training days. To run MIND-200K, choose `--dataset MIND-200K` and a new work directory on the first command. The pipeline uses the labeled Large archives, selects 200,000 users, and saves the sample and seed. Subsequent commands for that directory reuse the recorded dataset.

See [local archive inputs](FROM_RAW_MIND.md), [candidate construction](CONSISTENT_CANDIDATES.md), and [asset preparation](ASSET_PREPARATION.md).

## 3. Train and evaluate

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --execute --resume
~~~

Choose `--models NAML`, `--models all5`, or `--models all6` when creating the study. All6 runs six models x IMP/RandS50 training x five test protocols: 12 trainings/60 evaluations.

| Training | Development | Test protocols |
| --- | --- | --- |
| IMP | IMP | IMP, RandS30/90/150, full Total |
| RandS50 | RandS50 | IMP, RandS30/90/150, full Total |

Defaults are batch size 16, learning rate 1e-4, four optimizer negatives, minimum 5/maximum 50 epochs, patience 5, and min_delta 1e-4. [Development settings](DEVELOPMENT.md) define checkpoint selection. Test metrics do not choose settings.

To run every stage with one command, use a new work directory:

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small_all" --models all6 --download --accept-mind-license --execute
~~~

## 4. Inspect or resume

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --status
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --check-only
~~~

A source snapshot is saved under `<WORK>/runtime`. After a checkout update, resume from that snapshot:

~~~cmd
RUN_EXPERIMENT.cmd "../tors_runs/mind_small/runtime/scripts/run_mind_pipeline.py" --work-dir "../tors_runs/mind_small" --execute --resume
~~~

Source, input hashes, settings, seed, optimizer/RNG state, and stopping state must match. Completed jobs and committed compatible evaluation chunks are reused. Changed training data/configuration requires a new study directory.

| Mode | Effect |
| --- | --- |
| `--dry-run` | Preview without writes, downloads, model imports, or seed selection |
| `--prepare-only` | Download/register inputs; build masters, proxy, and candidates |
| `--check-only` | Validate recorded input/source/candidate fingerprints |
| `--execute` | Prepare missing declared inputs, train, evaluate, and summarize |
| `--status` | Read saved state |
| `--resume` | Continue the recorded study |

## Study seed and reproduction

`--seed random` is the new-study default. One unsigned 32-bit seed is selected when sealing the plan and saved in `control/pipeline_plan.json`. Helpers, paired IMP/RandS conditions, and models receive it. Per-row sampling streams are deterministic.

The seed is not redrawn per epoch, model, evaluation, or resume. `--seed <INTEGER>` records a replay seed for a new study; valid integers are 0 through 4294967295. A different resume seed is rejected. Hardware and library versions are also part of reproducibility.

Table3 uses a separately derived, recorded analysis seed. Every model shares each resampled request subset. These are not independent model trainings.

## 5. Read the results

~~~text
<WORK>/
  downloads/                         Original ZIPs and receipts
  assets/                            Registered pretrained/DIGAT assets
  masters/                           Chronological IMP splits
  dataset/                           Proxy, candidates, and provenance
  runtime/                           Frozen source
  control/
    pipeline_plan.json               Resolved options and selected seed
    experiment_plan.json             Jobs and pinned inputs
    pipeline_state.json              Saved execution state
    completion.json                  Completion record
  runs/<MODEL>_<TRAIN>_seed<SEED>/
    experiment.json                  Native configuration and selected checkpoint
    evaluation/                      Development/test scores and metrics
  results/
    config.json                      Study and per-run settings
    metrics.csv                      Accuracy by model/training/dev/evaluation
    timings_net.csv                  Rest-adjusted times
    tables/                          All-six-model tables and validation
~~~

After completion, CSV `configuration_id`/`configuration_file` fields identify entries in `results/config.json`. Native manifests retain configuration, checkpoint, source, and input fingerprints.

For all6, `results/tables/` includes `table2_summary.csv`, `table3_summary.csv`, `table5_summary.csv`, per-seed tables, resampling metadata, timing CSVs, and `validation.json`. Table2 reports metrics/model ranks; Table3 compares model-pair decisions with Total; Table5 compares both trainings on the same full Total.

[Manuscript reference values](PAPER_RESULTS.md) remain separate. Across-seed SD is NA for one training seed.

## Evaluation and aggregation

AUC, MRR, nDCG@5, and nDCG@10 use a 0-1 scale. IMP entries can have several positives. MRR averages reciprocal positive ranks within each entry; nDCG uses its cutoff and ideal multi-positive ranking.

The primary native-entry mean equally weights IMP impressions or RandS/Total positive entries. Request-macro sensitivity first averages within a common user/time/history request. Both aggregations are retained in table CSVs.

Exact score ties use `stable_original_candidate_order`. RandS/Total store the positive first, so tied entries can favor that position. Metric files record the policy and retain raw scores. An ordering-dependent tie is not a strict score preference.

## Resources and progress

The terminal displays total/current-epoch training and current-test progress. Redirected logs retain measured snapshots and hourly summaries.

Safe-boundary rests default to 120 seconds after 600 active training seconds and 900 seconds per 10,800 elapsed seconds. Actual rests are separately recorded and subtracted once from net timing. Preparation, development, tests, and total runner time are distinct.

GloVe archives/extracted text and parsed caches require several GB. Full Total score storage scales with all eligible request/candidate pairs. Allow additional disk for checkpoints, semantic graph assets, and per-run outputs. Runtime and memory requirements depend on hardware; use recorded run measurements for estimates.

