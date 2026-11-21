# MIND-Small and MIND-200K data preparation

Use the main pipeline to create a study from original MIND archives. It registers pretrained files, preserves master requests, builds the publication-time proxy and candidates, and seals the inputs.

## Download original archives

Read the [MIND terms](https://msnews.github.io/). Original-data download URLs follow the [Microsoft Recommenders MIND loader](https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py).

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --models all6 --download --accept-mind-license --prepare-only
~~~

ZIPs and source/size/SHA-256 receipts remain in `<WORK>/downloads`. `--mind-train-sha256` and `--mind-dev-sha256` additionally check supplied trusted checksums.

## Choose the dataset

| Dataset option | Source ZIPs | Requests used for the study |
| --- | --- | --- |
| `MIND-small` (default) | `MINDsmall_train.zip`, `MINDsmall_dev.zip` | All labeled requests |
| `MIND-200K` | `MINDlarge_train.zip`, `MINDlarge_dev.zip` | All labeled requests from 200,000 sampled users |

The 200K sampler draws users uniformly without replacement from the distinct IDs in the two labeled archives. A user who appears in both belongs to the sampling population once. The same cohort is used throughout training, validation, and testing. The official unlabeled MIND-Large test archive is not an input. A new study draws a seed by default; pass `--seed <INTEGER>` with the same archives to repeat a draw.

The master directory records the selected IDs in `<WORK>/masters/provenance/sample_users.json`. Its manifest records the population size, sample size, seed and file hash. The sampled IDs and downloaded data stay in the study directory. The published paper's MIND-200K reference tables describe its historical cohort; a new draw may produce different statistics and metrics.

## Use local files

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --models all6 --train-zip "../downloads/MINDsmall_train.zip" --dev-zip "../downloads/MINDsmall_dev.zip" --glove-archive "../downloads/glove.6B.zip" --sag-model-source "../downloads/all-mpnet-base-v2" --prepare-only
~~~

~~~cmd
RUN_MIND_PIPELINE.cmd --dataset MIND-200K --work-dir "../tors_runs/mind_200k" --models all6 --train-zip "../downloads/MINDlarge_train.zip" --dev-zip "../downloads/MINDlarge_dev.zip" --glove-archive "../downloads/glove.6B.zip" --sag-model-source "../downloads/all-mpnet-base-v2" --prepare-only
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_200k" --execute --resume
~~~

NAML/all5 require no MPNet source. `--glove-text` accepts the 300d text; `--embedding-root` accepts registered GloVe assets. See [asset registration](ASSET_PREPARATION.md).

## Generated dataset

Dates are checked, then assigned to train November 9-13, dev November 14, and test November 15. User/time/normalized-full-history groups remain intact.

The proxy is each article's earliest clicked or unclicked impression. MIND-Small uses its supplied masters. MIND-200K scans both complete labeled Large archives, so removing users from the study does not move an article's first observation forward. History-only or metadata-only articles receive no invented timestamp.

~~~text
<WORK>/dataset/
  manifest.json
  consistent_candidate_manifest.json
  news_publish_time.tsv
  exclusions.jsonl
  provenance/
    observed_first_impression_min.json
    full_total_manifest.json
    PREPARATION_VERIFICATION.json
  train/
    behaviors.tsv
    behaviors_rs50.tsv
    news.tsv
    entity_embedding.vec
  dev/
    behaviors.tsv
    behaviors_rs50.tsv
    news.tsv
    entity_embedding.vec
  test/
    behaviors.tsv
    behaviors_imp_eligible.tsv
    behaviors_rs30.tsv
    behaviors_rs90.tsv
    behaviors_rs150.tsv
    behaviors_total.tsv
    news.tsv
    entity_embedding.vec
~~~

The selected paper encoders do not use NNR context embeddings; their explicit provenance mode omits those artifacts and preserves official entity data. `--wikidata-graph <ZIP_OR_TSV>` instead builds real context features for encoders requiring them.

[Candidate construction](CONSISTENT_CANDIDATES.md) specifies strict time boundaries, exclusions, sampling, and target preservation. Empty-negative requests remain in masters; derived exclusions are counted and recorded. Final test protocols share eligible target support.

## Verify and execute

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --check-only
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --execute --resume
~~~

Preparation records the random seed, sources, inputs, and model list. All6 execution produces 12 trainings/60 evaluations and Tables2/3/5. See [Usage](USAGE.md) for output/configuration files and resume commands.

This five-day training split reserves one day for development; manuscript reference tables use six training days. Dataset rows, pretrained weights, and checkpoints remain outside Git.

