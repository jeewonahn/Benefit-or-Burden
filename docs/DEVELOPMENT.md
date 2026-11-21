# Development and model settings

Development follows the training protocol. Final tests are evaluated after selecting the development-best checkpoint.

| Training | Development | File |
| --- | --- | --- |
| IMP | IMP | `dev/behaviors.tsv` |
| RandS50 | RandS50 | `dev/behaviors_rs50.tsv` |

The raw-MIND workflow uses train November 9-13 and dev November 14. Request groups are disjoint. Fixed dev candidates follow the [common candidate rule](CONSISTENT_CANDIDATES.md).

## Training settings

| Setting | Released value |
| --- | --- |
| Models | NAML, NRMS, LSTUR, NPA, CNE-SUE, DIGAT |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Optimizer negatives per positive | 4 |
| Minimum/maximum epochs | 5/50 |
| Patience | 5 |
| Significant improvement | 0.0001 |
| GloVe | 6B 300d |
| Seed | Randomly selected once and recorded |

The RandS50 file has up to 50 unique negatives per positive; the optimizer separately samples four slots. Runners record effective settings. Upstream defaults can differ from this released configuration.

For MIND-200K, the NNR runner selects its upstream `200k` preset (dropout 0.2 and four GCN layers). The explicit epoch and development settings above still apply. DIGAT uses the prepared cohort paths and records the study dataset separately from its upstream `MIND-small` key, which selects the labeled-test format and cache filenames.

## Checkpoint selection

Each epoch evaluates the matching dev file with:

~~~text
(AUC + MRR + (nDCG@5 + nDCG@10) / 2) / 3
~~~

The highest finite score selects the best checkpoint. Patience resets when the score exceeds the significant-improvement anchor by more than 0.0001. Best selection and significant-improvement counting are separate. Minimum training epochs do not prevent selecting an earlier checkpoint. Reaching the epoch limit does not establish convergence.

Each run saves dev metrics/decisions, effective settings, input identities, optimizer/RNG/stopping state, and the selected model. Restart requires these same recorded inputs and settings. Test metrics do not select epochs, hyperparameters, or seeds.

## Model computation

Implementations build on [NNR](https://github.com/Veason-silverbullet/NNR) and [DIGAT](https://github.com/Veason-silverbullet/DIGAT), retaining their licenses.

NPA aligns the personalized query with each user/article pair. CNE aligns cross-view cell states by original article identity before gating. These recorded implementation corrections add no trainable parameters but change batch assignment relative to the uncorrected source.

NAML/NRMS/LSTUR reuse compatible news/user representations. NPA retains personalized news attention, CNE-SUE candidate-conditioned attention, and DIGAT dual-graph interaction. Native/cached comparisons document the tested cases; input/checkpoint fingerprints guard reuse.

Vocabulary, article metadata, and DIGAT graph preparation use text from supplied splits. This is a transductive text/metadata setting; dev/test impression labels are not optimizer targets. See [metric conventions](USAGE.md#evaluation-and-aggregation).

