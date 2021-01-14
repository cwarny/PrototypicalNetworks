# Prototypical Networks for NLU

Adaptation of [this paper](https://arxiv.org/pdf/1703.05175.pdf) to the NLU use case.

## Prepare data

* `python scripts/prepare_data.py --data-dir data --input <tsv file>`
* input tsv file should be in the classic Alexa 3-col format (domain, intent, anno)

## Train model

* `python scripts/train_model.py --data-dir <data dir with the train.tsv, valid.tsv, test.tsv files>`

## Run tests

* `pytest`