# Prototypical Networks for NLU

## Prepare data

* `python scripts/prepare_data.py --data-dir data --input <tsv file>`
* input tsv file should be in the classic Alexa 3-col format (domain, intent, anno)

## Train model

* `python scripts/train_model.py`

## Run tests

* `pytest`