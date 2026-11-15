# Benefit or Burden?: Using Impression Logs for News Recommendation Systems

Code for the ACM Transactions on Recommender Systems manuscript by **Jeewon Ahn, Hong-Kyun Bae, and Sang-Wook Kim**. The manuscript is under revision.

We compare IMP, randomly sampled (RandS), and full Total evaluation, and IMP versus RandS training, using NAML, NRMS, LSTUR, NPA, CNE-SUE, and DIGAT.

Built on [NNR](https://github.com/Veason-silverbullet/NNR) and [DIGAT](https://github.com/Veason-silverbullet/DIGAT). Original licenses are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 1. Installation

Use **Python 3.9**. The requirements target PyTorch 1.12.1/CUDA 11.3 with matching torchtext and torch-scatter binaries.

~~~cmd
git clone https://github.com/jeewonahn/Benefit-or-Burden.git
cd Benefit-or-Burden
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
set "TORS_PYTHON=%CD%\.venv\Scripts\python.exe"
~~~

On Linux, activate with `source .venv/bin/activate` and use `python -B -u scripts/run_mind_pipeline.py` in place of the CMD entry point. Match binary packages to the selected platform; see [Usage](docs/USAGE.md).

## 2. Download and preprocess MIND

Read the [MIND terms](https://msnews.github.io/). Select a new work directory **outside the repository**, then prepare all inputs:

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --models all6 --download --accept-mind-license --prepare-only
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --check-only
~~~

Preparation performs these steps:

1. Download `MINDsmall_train.zip` and `MINDsmall_dev.zip` using the URLs maintained by [Microsoft Recommenders](https://github.com/recommenders-team/recommenders/blob/main/recommenders/datasets/mind.py).
2. Register GloVe **6B 300d** and the pinned MPNet files for DIGAT.
3. Preserve complete request groups and split original dates into **train November 9-13, dev November 14, and test November 15**.
4. Create `news_publish_time.tsv` from the earliest clicked or unclicked impression of each article.
5. Generate IMP/RandS50 train/dev files and IMP, RandS30/90/150, and full Total test candidates.

The publication-time value is a proxy. Negatives satisfy `t - 24h < proxy < t`, exclude request history and all current positives, and are sampled without replacement. The clicked target is retained independently of the negative window. See [candidate construction](docs/CONSISTENT_CANDIDATES.md).

This workflow uses five training days and a held-out development day; the manuscript reference tables use six training days. MIND-200K is a separate population and is not generated from the small archives. [Local archive inputs](docs/FROM_RAW_MIND.md) are supported.

## 3. Train and evaluate

~~~cmd
RUN_MIND_PIPELINE.cmd --work-dir "../tors_runs/mind_small" --execute --resume
~~~

Models run in the order NAML, NRMS, LSTUR, NPA, CNE-SUE, and DIGAT. DIGAT's shared semantic graph is built before its first training job.

| Training | Development for checkpoint selection | Final evaluations |
| --- | --- | --- |
| IMP | IMP | IMP, RandS30/90/150, full Total |
| RandS50 | RandS50 | IMP, RandS30/90/150, full Total |

`all6` produces **12 trainings and 60 evaluations**. Choose `--models NAML` for NAML only or `--models all5` for the NNR-derived models on the first preparation command.

Defaults are batch size 16, learning rate 0.0001, four optimizer negatives, minimum 5/maximum 50 epochs, patience 5, and min-delta 0.0001. [Development settings](docs/DEVELOPMENT.md) specify the weighted selection score and model computation.

A study randomly selects one seed, records it, and shares it across paired comparisons and candidate preparation. Resume retains the seed, optimizer/RNG state, completed jobs, and compatible evaluation chunks. `--seed <INTEGER>` selects an explicit replay seed for a new study.

After a checkout update, resume with the study's frozen source:

~~~cmd
RUN_EXPERIMENT.cmd "../tors_runs/mind_small/runtime/scripts/run_mind_pipeline.py" --work-dir "../tors_runs/mind_small" --execute --resume
~~~

## 4. Results and configuration

After completion, the work directory contains:

~~~text
results/
  config.json                   Resolved settings, actual seed, and run identities
  metrics.csv                   Metrics by model, training/dev, and evaluation
  timings_net.csv               Measured times with rests excluded
  tables/                       Generated for all6
    table2_summary.csv
    table3_summary.csv
    table5_summary.csv
    validation.json
~~~

CSV rows link to `config.json` through `configuration_id` and `configuration_file`. Native run manifests retain effective settings, selected checkpoints, and input/source fingerprints. [Usage](docs/USAGE.md#evaluation-and-aggregation) defines metric aggregation and exact-score tie handling.

New measurements remain separate from the manuscript tables below. A single training seed has no across-seed standard deviation. Table3's shared request resamples are not additional trainings.


## Results Reported in the Paper

The following tables contain **paper-reported reference results**, not newly measured results from this checkout. We preserve the manuscript's printed precision, ranks, and values. AUC, MRR, and nDCG use a 0–1 scale; Table 3 uses percentages. MIND-Small and MIND-200K are listed separately. Complete tables and source notes are available in [Paper results](docs/PAPER_RESULTS.md).

### Table 2 — model rankings under different evaluation protocols

Each cell is **rank (metric value)**. RandS uses k=30. Table 2 contains AUC, MRR and nDCG@5; its surrounding text does not explicitly identify one training profile for these rows. No nDCG@10 or training label is inferred.

<details>
<summary>Show Table 2 for MIND-Small and MIND-200K</summary>

### MIND-Small

| Metric | Model | IMP test | Total test | RandS test (k=30) |
| --- | --- | --- | --- | --- |
| AUC | CNE-SUE | 1 (.669) | 1 (.880) | 1 (.880) |
| AUC | DIGAT | 2 (.667) | 2 (.857) | 2 (.857) |
| AUC | NAML | 5 (.646) | 3 (.853) | 3 (.853) |
| AUC | LSTUR | 3 (.654) | 4 (.834) | 4 (.834) |
| AUC | NRMS | 4 (.652) | 5 (.816) | 5 (.816) |
| AUC | NPA | 6 (.620) | 6 (.809) | 6 (.809) |
| MRR | CNE-SUE | 2 (.329) | 1 (.066) | 1 (.506) |
| MRR | DIGAT | 3 (.328) | 2 (.048) | 2 (.492) |
| MRR | NAML | 1 (.349) | 3 (.043) | 3 (.469) |
| MRR | LSTUR | 5 (.320) | 4 (.038) | 4 (.452) |
| MRR | NRMS | 4 (.324) | 5 (.036) | 5 (.443) |
| MRR | NPA | 6 (.296) | 6 (.028) | 6 (.434) |
| nDCG@5 | CNE-SUE | 2 (.344) | 1 (.056) | 1 (.530) |
| nDCG@5 | DIGAT | 3 (.344) | 2 (.042) | 2 (.524) |
| nDCG@5 | NAML | 1 (.353) | 3 (.039) | 3 (.497) |
| nDCG@5 | LSTUR | 5 (.334) | 4 (.033) | 4 (.478) |
| nDCG@5 | NRMS | 4 (.339) | 5 (.032) | 5 (.460) |
| nDCG@5 | NPA | 6 (.310) | 6 (.021) | 6 (.442) |

### MIND-200K

| Metric | Model | IMP test | Total test | RandS test (k=30) |
| --- | --- | --- | --- | --- |
| AUC | CNE-SUE | 2 (.653) | 1 (.871) | 1 (.871) |
| AUC | DIGAT | 1 (.665) | 2 (.858) | 2 (.858) |
| AUC | NAML | 5 (.612) | 3 (.850) | 3 (.850) |
| AUC | LSTUR | 3 (.633) | 4 (.830) | 4 (.830) |
| AUC | NRMS | 4 (.631) | 5 (.814) | 5 (.814) |
| AUC | NPA | 6 (.606) | 6 (.808) | 6 (.808) |
| MRR | CNE-SUE | 3 (.317) | 1 (.064) | 1 (.514) |
| MRR | DIGAT | 1 (.345) | 2 (.050) | 2 (.491) |
| MRR | NAML | 2 (.322) | 3 (.047) | 3 (.472) |
| MRR | LSTUR | 5 (.302) | 4 (.046) | 4 (.472) |
| MRR | NRMS | 4 (.306) | 5 (.036) | 5 (.466) |
| MRR | NPA | 6 (.283) | 6 (.027) | 6 (.465) |
| nDCG@5 | CNE-SUE | 2 (.330) | 1 (.056) | 1 (.538) |
| nDCG@5 | DIGAT | 1 (.333) | 2 (.044) | 2 (.520) |
| nDCG@5 | NAML | 3 (.322) | 3 (.038) | 3 (.501) |
| nDCG@5 | LSTUR | 5 (.313) | 4 (.037) | 4 (.493) |
| nDCG@5 | NRMS | 4 (.317) | 5 (.034) | 5 (.489) |
| nDCG@5 | NPA | 6 (.294) | 6 (.023) | 6 (.483) |

Source: Section 6.2 RQ1–RQ2, Table 2 (`Tables/Table2.tex`). [CSV with source line numbers](docs/paper_reported_results/paper_table2_rankings.csv).

</details>

### Table 3 — matching rates against Total test

The paper compares superiority decisions for 15 model pairs over ten sampled test-entry sets. Higher matching rates mean closer agreement with Total-test model comparisons; they do not mean higher recommendation accuracy.

| Dataset | Evaluation | AUC (%) | MRR (%) | nDCG@5 (%) | nDCG@10 (%) |
| --- | --- | --- | --- | --- | --- |
| MIND-Small | IMP | 70.7 | 69.3 | 69.3 | 68.0 |
| MIND-Small | RandS30 | 100.0 | 93.8 | 92.0 | 93.3 |
| MIND-Small | RandS90 | 100.0 | 94.2 | 92.6 | 93.5 |
| MIND-Small | RandS150 | 100.0 | 94.8 | 93.5 | 94.3 |
| MIND-200K | IMP | 68.9 | 74.7 | 76.4 | 71.1 |
| MIND-200K | RandS30 | 100.0 | 92.4 | 91.1 | 92.0 |
| MIND-200K | RandS90 | 100.0 | 93.6 | 92.0 | 92.8 |
| MIND-200K | RandS150 | 100.0 | 94.3 | 92.7 | 93.7 |

### Table 5 — IMP versus RandS training, evaluated on Total

Each arrow shows **IMP training → RandS training** under the same full Total evaluation. In the paper's reported cells, RandS training is higher for every listed model and metric. [The full source table](docs/PAPER_RESULTS.md#table-5-imp-versus-rands-training-evaluated-using-total-test) also retains the printed gain percentages.

| Dataset | Model | AUC: IMP → RandS | MRR: IMP → RandS | nDCG@5: IMP → RandS | nDCG@10: IMP → RandS |
| --- | --- | --- | --- | --- | --- |
| MIND-Small | CNE-SUE | 0.7056 → 0.8712 | 0.0361 → 0.0645 | 0.0320 → 0.0562 | 0.0382 → 0.0661 |
| MIND-Small | DIGAT | 0.7001 → 0.8582 | 0.0353 → 0.0498 | 0.0300 → 0.0439 | 0.0371 → 0.0503 |
| MIND-Small | NAML | 0.6836 → 0.8495 | 0.0339 → 0.0467 | 0.0302 → 0.0396 | 0.0354 → 0.0470 |
| MIND-Small | LSTUR | 0.6916 → 0.8296 | 0.0400 → 0.0459 | 0.0368 → 0.0378 | 0.0420 → 0.0464 |
| MIND-Small | NRMS | 0.6929 → 0.8139 | 0.0284 → 0.0361 | 0.0261 → 0.0339 | 0.0300 → 0.0392 |
| MIND-Small | NPA | 0.7087 → 0.8076 | 0.0316 → 0.0326 | 0.0277 → 0.0290 | 0.0338 → 0.0389 |
| MIND-200K | CNE-SUE | 0.6759 → 0.8795 | 0.0305 → 0.0663 | 0.0275 → 0.0561 | 0.0330 → 0.0666 |
| MIND-200K | DIGAT | 0.6971 → 0.8572 | 0.0320 → 0.0482 | 0.0296 → 0.0419 | 0.0356 → 0.0481 |
| MIND-200K | NAML | 0.6657 → 0.8532 | 0.0260 → 0.0434 | 0.0227 → 0.0388 | 0.0265 → 0.0443 |
| MIND-200K | LSTUR | 0.6611 → 0.8345 | 0.0307 → 0.0383 | 0.0276 → 0.0298 | 0.0304 → 0.0378 |
| MIND-200K | NRMS | 0.6829 → 0.8164 | 0.0185 → 0.0359 | 0.0136 → 0.0317 | 0.0179 → 0.0363 |
| MIND-200K | NPA | 0.6892 → 0.8090 | 0.0200 → 0.0285 | 0.0160 → 0.0212 | 0.0202 → 0.0300 |

Dataset statistics and the paper's NAML evaluation times are in [Tables 1 and 4](docs/PAPER_RESULTS.md). Those evaluation times are not training times or runtime estimates for another computer. Table-level source ambiguities are recorded in [source notes](docs/PAPER_RESULTS.md#source-notes-and-unresolved-provenance); values have not been silently reconciled between tables.

## Cached Inference and Progress Display

Compatible representations are reused for each checkpoint. NPA personalized attention, CNE-SUE candidate-conditioned attention, and DIGAT dual-graph interactions are retained. Native/cached comparisons check the evaluated cases. NPA/CNE batch-alignment corrections are recorded per run.

The terminal displays total/current-epoch training progress and current evaluation progress. Training, development, evaluation, preparation, and measured rest have separate timing records. See [Usage](docs/USAGE.md).

## Code Layout

~~~text
NNR-main/                  NNR-derived model source and license
DIGAT/                     DIGAT source, provenance, and license
scripts/                  Preparation, training, evaluation, and tables
tests/                    Software correctness tests
RUN_MIND_PIPELINE.cmd      Download-to-results entry point
RUN_EXPERIMENT.cmd         Individual Python-tool entry point
requirements.txt          Compatible Python/CUDA package profile
docs/USAGE.md             Step-by-step execution and outputs
docs/PAPER_RESULTS.md     Manuscript reference tables
~~~

## Citation

~~~bibtex
@article{ahn_benefit_or_burden,
  author  = {Jeewon Ahn and Hong-Kyun Bae and Sang-Wook Kim},
  title   = {{Benefit or Burden?}: Using Impression Logs for News Recommendation Systems},
  journal = {ACM Transactions on Recommender Systems},
  note    = {Manuscript under revision}
}
~~~

See [CITATION.bib](CITATION.bib). Publication metadata will be added when assigned.

