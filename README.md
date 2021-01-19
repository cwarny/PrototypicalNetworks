# Prototypical Networks for NLU

Adaptation of [this paper](https://arxiv.org/pdf/1703.05175.pdf) to the NLU use case.

A prototypical network is trained in a series of "training episodes". An episode consists in a mini-classification problem where you are given:

* an N-way classification problem
* a couple of labelled samples for each of the N classes in that episode ("support vectors")

Then the network's task is to classify a couple of unlabelled samples ("query vectors"). 

In each episode, you randomly select N classes (from a larger set of classes) to create this N-way classification problem. What the prototypical network learns is to do this kind of "on-the-fly" N-way classification based on a small number of M support vectors. In a nutshell, the prototypical network learns to do *on-the-fly N-way classification using M-shot learning*.

This means that the classes a prototypical network is applied on at inference time *do not need to be the same classes it was trained on*. What the network learns is to solve a classification problem based on a set of labelled examples provided on the fly. It can therefore accommodate new classes not seen in training. 

This technique is useful when we have very limited training data.

## Prepare data

* `python scripts/prepare_data.py --data-dir data --input <tsv file>`
* input tsv file should be in the classic Alexa 3-col format (domain, intent, anno)

## Train model

* `python scripts/train_model.py --data-dir <data dir with the train.tsv, valid.tsv, test.tsv files>`

## Run tests

* `pytest`