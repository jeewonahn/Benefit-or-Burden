# Pretrained assets

The main pipeline downloads/registers files during `--prepare-only`. All six models use **GloVe 6B 300d**; DIGAT also uses pinned **all-mpnet-base-v2**. Assets stay outside Git.

## GloVe 6B 300d

Source: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/).

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_model_assets.py glove --download --output "../model_assets/glove6b"
~~~

Local inputs:

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_model_assets.py glove --archive "../downloads/glove.6B.zip" --output "../model_assets/glove6b"
RUN_EXPERIMENT.cmd scripts/prepare_model_assets.py glove --text "../downloads/glove.6B.300d.txt" --output "../model_assets/glove6b_text"
~~~

Outputs are `GLOVE_ASSET_MANIFEST.json` and the verified 300d text. Registration checks 400,000 ordinary rows, an optional final `<unk>` row, finite vector components, archive integrity, and hashes. `--expected-sha256` checks an external input checksum. `--check-only --output <DIRECTORY>` verifies registered assets.

The released 6B configuration differs from upstream DIGAT's 840B 300d default. Use the recorded source/dimensions for comparable runs.

## DIGAT semantic model

Source: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).

Pinned revision: `e8c3b32edf5434bc2275fc9bab85f82640a19130`.

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_model_assets.py semantic --download --output "../model_assets/all-mpnet-base-v2"
~~~

Or register a local snapshot:

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_model_assets.py semantic --source "../downloads/all-mpnet-base-v2" --revision e8c3b32edf5434bc2275fc9bab85f82640a19130 --output "../model_assets/all-mpnet-base-v2"
~~~

`TORS_ASSET_MANIFEST.json` records revision/configuration/tokenizer/weight hashes. Registration verifies file structure without executing inference. Semantic vectors have 768 dimensions.

## DIGAT shared preprocessing

All6 execution builds shared metadata and the semantic article graph before the first DIGAT job. An immutable `control/digat_asset_seal.json` pins six shared artifacts. Each run retains its own selected user/history data.

For a standalone build:

~~~cmd
RUN_EXPERIMENT.cmd scripts/prepare_digat_assets.py --data-root "<WORK>/dataset" --embedding-root "<WORK>/assets/glove6b" --sag-model "<WORK>/assets/all-mpnet-base-v2" --output "<NEW_DIGAT_ASSETS>" --seed <RECORDED_STUDY_SEED>
~~~

`--check-only` verifies prerequisites without graph construction. Construction needs the compatible PyTorch/CUDA environment. Use the study's recorded seed; the main pipeline supplies it automatically.

## Reuse and licensing

Registered files are rehashed before reuse. Changed source/input/configuration requires distinct outputs. `--resume` can continue matching DIGAT preparation.

Obtain MIND, GloVe, and pretrained models under their own terms and retain their notices when sharing assets. This repository distributes code and source licenses, not dataset rows, pretrained weights, or trained checkpoints.

