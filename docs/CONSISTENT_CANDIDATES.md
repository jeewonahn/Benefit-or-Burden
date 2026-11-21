# Candidate construction

`scripts/candidate_policy.py` defines `complete_proxy_strict_request_consistent_v1`. The generator writes a new dataset and the validator checks its recorded inputs and outputs.

## Publication-time proxy

For each article, use the minimum timestamp across clicked and unclicked impression candidates. MIND-Small uses the IMP masters. MIND-200K uses both complete labeled MIND-Large source archives, even when the earliest observation was made by a user outside the sampled cohort. Only the 200,000 selected users contribute training or evaluation requests. `news_publish_time.tsv` records the proxy. History-only or metadata-only appearances receive no invented timestamp.

It is first-observed impression time, not publisher time. Dataset provenance preserves source coverage, original clock convention, and the observed minimum map.

## Eligible negatives

For a request at time `t`:

1. Keep articles with `t - 24h < proxy_time < t`; both boundaries are strict.
2. Remove the request's recorded history.
3. Remove all positives in the same user/time/normalized-full-history group, including duplicate source rows.
4. Retain the target positive independently of the negative time filter.

Later user clicks are not exclusions. Empty-negative requests remain in masters and are excluded from derivatives with reasons/counts. All final test protocols share eligible request/target support.

## Sampling

| Role | Candidates |
| --- | --- |
| IMP train/dev | Logged impression candidates |
| RandS50 train/dev | One positive plus up to50distinct uniform negatives |
| RandS30/90/150 test | One positive plus nested 30/90/150-negative prefixes |
| Full Total test | One positive plus every eligible negative |

Sampling is without replacement with `min(k, pool_size)`. Streams derive from seed/split/source row/positive index. Samples remain fixed across models and epochs, preserving row order and positive multiplicity.

The optimizer's four negative slots are separate from the candidate-file pool. Development ranks its full fixed file. See [development settings](DEVELOPMENT.md).

## Data and certificates

The main pipeline generates masters and:

- `train/behaviors_rs50.tsv` and `dev/behaviors_rs50.tsv`.
- `test/behaviors_imp_eligible.tsv`, `behaviors_rs30.tsv`, `behaviors_rs90.tsv`, `behaviors_rs150.tsv`, and `behaviors_total.tsv`.
- `news_publish_time.tsv`, `exclusions.jsonl`, and `consistent_candidate_manifest.json`.
- `provenance/observed_first_impression_min.json`, `full_total_manifest.json`, and `PREPARATION_VERIFICATION.json`.

For independently supplied complete IMP masters:

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_consistent_candidates.py --source-data-root "<IMP_MASTER_ROOT>" --output "<NEW_DATASET_ROOT>" --seed <RECORDED_STUDY_SEED>
~~~

The certificate binds source masters, proxy/minimum map, exclusions, policy, seed, and candidate hashes. The Total descriptor binds the full eligible candidate file and dataset identity. Changed inputs or incompatible certificates are rejected.

The selected paper encoders omit unused context artifacts and retain official entity assets. Context-consuming encoders require real context inputs. See [data layout](FROM_RAW_MIND.md) and [metric aggregation/ties](USAGE.md#evaluation-and-aggregation).

