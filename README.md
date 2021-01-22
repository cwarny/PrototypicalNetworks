# Prototypical Networks for NLU

Adaptation of [this paper](https://arxiv.org/pdf/1703.05175.pdf) to the NLU use case.

A prototypical network is trained in a series of "training episodes". An episode consists in a mini-classification problem where you are given:

* an N-way classification problem
* a couple of labelled samples for each of the N classes in that episode ("support vectors")

Then the network's task is to classify a couple of unlabelled samples ("query vectors"). 

In each episode, you randomly select N classes (from a larger set of classes) to create this N-way classification problem. What the prototypical network learns is to do this kind of "on-the-fly" N-way classification based on a small number of M support vectors. In a nutshell, the prototypical network learns to do *on-the-fly N-way classification using M-shot learning*.

This means that the classes a prototypical network is applied on at inference time *do not need to be the same classes it was trained on*. What the network learns is to solve a classification problem based on a set of labelled examples provided on the fly. It can therefore accommodate new classes not seen in training. 

This technique is useful when we have very limited training data.

## Setup

* `source activate pytorch_p36`
* `cd <this repo>`
* `python setup.py install`
* `pip install -r requirements.txt`

## Prepare data

* `python scripts/prepare_data.py --data-dir data --input <tsv file>`
* Input tsv file should be in the classic Alexa 3-col format (domain, intent, anno)
* Example tsv file can be found here: `/apollo/env/HoverboardDefaultMLPS3Tool/bin/mlps3 cp -k "com.amazon.snl-3pi18n.team.hoverboard" s3://hoverboard-shared-snl-3pi18n-eu-west-1/nlu/datasets/fud.tsv.tgz .`

## Train model

* `/apollo/env/HoverboardDefaultMLPS3Tool/bin/mlps3 cp -r s3://blu-core-model-training-eu/snl/models/huggingface ~/.cache`
* Updated config file: `emacs <yaml config file>`
* `python scripts/train_model.py -c <yaml config file>`

## Run tests

* `pytest`