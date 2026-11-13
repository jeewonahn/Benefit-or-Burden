# Paper-reported results

These values are transcribed from the authors' manuscript. They are paper-reported references, not new measurements from this checkout. The original table precision, rankings and gain rows are retained. The source table filenames and SHA-256 hashes are recorded in [provenance.json](paper_reported_results/provenance.json).

MIND-Small and MIND-200K are separate datasets. Total denotes the full eligible candidate set. IMP and RandS can describe either training or evaluation; each table identifies the role. The paper uses RandS evaluation k=30 except when varying k in Table 3.

AUC, MRR and nDCG values use a 0-1 scale. Table 3 and Table 5 gain rows use percentages; Table 4 uses seconds.

## Table 1: dataset statistics

| Dataset | Users | Items | Clicks | Sparsity (%) |
| --- | --- | --- | --- | --- |
| MIND-Small | 94,057 | 65,238 | 2,385,357 | 99.96 |
| MIND-200K | 200,000 | 80,585 | 4,731,982 | 99.97 |

Source: Section 6.1, Table 1 (`Tables/Table1.tex`). [CSV](paper_reported_results/paper_table1_statistics.csv). These are the dataset statistics reported in the manuscript.

## Table 2: model rankings under different evaluation protocols

Each cell is **reported rank (reported metric value)**. `G@5` in the source is displayed as `nDCG@5`. The source table does not contain nDCG@10, so no nDCG@10 values are added. Its caption and surrounding text do not explicitly fix one training profile for these rows; do not infer that all rows are IMP-trained or reuse a Table 5 cell to fill this table. Underlining in the LaTeX is presentation markup and is not reproduced; the numeric ranks are unchanged.

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

Source: Section 6.2 RQ1–RQ2, Table 2 (`Tables/Table2.tex`). [CSV with source line numbers](paper_reported_results/paper_table2_rankings.csv).

## Table 3: matching rates against Total test

Matching rates (%) compare model-pair superiority decisions with the Total-test reference. Section 6.2 describes 15 model pairs and ten different sampled test-entry sets. The values below follow the manuscript table.

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

Source: Section 6.2 RQ2–RQ3, Table 3 (`Tables/Table3.tex`). [CSV](paper_reported_results/paper_table3_matching_rates.csv).

## Table 4: NAML evaluation time

| Model | Dataset | IMP test (s) | Total test (s) | RandS test (s) |
| --- | --- | --- | --- | --- |
| NAML | MIND-Small | 121.3 | 1165.7 | 141.1 |
| NAML | MIND-200K | 175.8 | 1754.2 | 204.5 |

Source: Section 6.2 RQ4, Table 4 (`Tables/Table4.tex`). [CSV](paper_reported_results/paper_table4_evaluation_seconds.csv). These are paper-reported **evaluation** times, not end-to-end training times or a runtime guarantee for the current code/hardware.

## Table 5: IMP versus RandS training, evaluated using Total test

Both training methods in this table are evaluated using the **same full Total protocol**, not IMP test for one and RandS test for the other. Metric rows use 0–1 units. Gain (%) rows are the paper's printed values and are not recomputed from the displayed rounded metric rows.

### MIND-Small

| Model | Training / reported gain | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| CNE-SUE | IMP | 0.7056 | 0.0361 | 0.0320 | 0.0382 |
| CNE-SUE | RandS | 0.8712 | 0.0645 | 0.0562 | 0.0661 |
| CNE-SUE | Gain (%) | 23.5 | 78.7 | 75.6 | 73.0 |
| DIGAT | IMP | 0.7001 | 0.0353 | 0.0300 | 0.0371 |
| DIGAT | RandS | 0.8582 | 0.0498 | 0.0439 | 0.0503 |
| DIGAT | Gain (%) | 22.6 | 41.1 | 46.3 | 35.6 |
| NAML | IMP | 0.6836 | 0.0339 | 0.0302 | 0.0354 |
| NAML | RandS | 0.8495 | 0.0467 | 0.0396 | 0.0470 |
| NAML | Gain (%) | 24.3 | 37.8 | 31.1 | 32.8 |
| LSTUR | IMP | 0.6916 | 0.0400 | 0.0368 | 0.0420 |
| LSTUR | RandS | 0.8296 | 0.0459 | 0.0378 | 0.0464 |
| LSTUR | Gain (%) | 20.0 | 14.8 | 2.7 | 10.5 |
| NRMS | IMP | 0.6929 | 0.0284 | 0.0261 | 0.0300 |
| NRMS | RandS | 0.8139 | 0.0361 | 0.0339 | 0.0392 |
| NRMS | Gain (%) | 17.5 | 27.1 | 30.0 | 30.7 |
| NPA | IMP | 0.7087 | 0.0316 | 0.0277 | 0.0338 |
| NPA | RandS | 0.8076 | 0.0326 | 0.0290 | 0.0389 |
| NPA | Gain (%) | 14.0 | 3.1 | 4.7 | 15.8 |

### MIND-200K

| Model | Training / reported gain | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| CNE-SUE | IMP | 0.6759 | 0.0305 | 0.0275 | 0.0330 |
| CNE-SUE | RandS | 0.8795 | 0.0663 | 0.0561 | 0.0666 |
| CNE-SUE | Gain (%) | 30.1 | 117.4 | 104.0 | 101.8 |
| DIGAT | IMP | 0.6971 | 0.0320 | 0.0296 | 0.0356 |
| DIGAT | RandS | 0.8572 | 0.0482 | 0.0419 | 0.0481 |
| DIGAT | Gain (%) | 23.0 | 50.6 | 41.6 | 35.1 |
| NAML | IMP | 0.6657 | 0.0260 | 0.0227 | 0.0265 |
| NAML | RandS | 0.8532 | 0.0434 | 0.0388 | 0.0443 |
| NAML | Gain (%) | 28.2 | 66.9 | 70.9 | 67.2 |
| LSTUR | IMP | 0.6611 | 0.0307 | 0.0276 | 0.0304 |
| LSTUR | RandS | 0.8345 | 0.0383 | 0.0298 | 0.0378 |
| LSTUR | Gain (%) | 26.2 | 24.8 | 8.0 | 24.3 |
| NRMS | IMP | 0.6829 | 0.0185 | 0.0136 | 0.0179 |
| NRMS | RandS | 0.8164 | 0.0359 | 0.0317 | 0.0363 |
| NRMS | Gain (%) | 19.5 | 94.1 | 133.1 | 102.8 |
| NPA | IMP | 0.6892 | 0.0200 | 0.0160 | 0.0202 |
| NPA | RandS | 0.8090 | 0.0285 | 0.0212 | 0.0300 |
| NPA | Gain (%) | 17.4 | 42.5 | 32.5 | 48.5 |

Source: Section 6.2 RQ6, Table 5 (`Tables/Table5.tex`). [CSV](paper_reported_results/paper_table5_training_comparison.csv).

## Source notes and unresolved provenance

- Table 3's MIND-Small IMP cells are 70.7, 69.3, 69.3 and 68.0 percent; their arithmetic mean is 69.325 percent. The RQ2 prose in `6_validation.tex` line 60 reports 71.1 percent for the across-metric IMP average without explicitly repeating the dataset scope. That value also rounds the combined eight IMP cells' mean of 71.05 percent, so it is not treated here as a proven arithmetic error. The table cells and prose value are retained with their different or unclear scopes; the source text has not been silently corrected.
- Table 2 and Table 5 report different values in some apparently related positions. Their training/checkpoint relationship is not established by the captions alone. Neither table is used to overwrite the other.
- The manuscript says six model names but contains the phrase "five state-of-the-art models" in Section 6.1. The tables contain six models; this documentation lists the actual names without editing the manuscript.
- These references do not supply the missing historical seeds, exact validation/checkpoint-selection configuration, original prediction files, or complete R2 experiment lineage. See [Usage](USAGE.md) for required run records and input verification.
- Do not overwrite experiment metrics with these reported reference values, treat them as a test oracle for fabricated outputs, or label them as newly reproduced results. Runtime outputs belong in separate run directories.
