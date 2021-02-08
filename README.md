# Prototypical Networks for NLU

This repo adapts [this paper](https://arxiv.org/pdf/1703.05175.pdf) to the NLU use case and expands in a few directions:

* We explore a more flexible definition of distance beyond Euclidean distance
* We explore alternative finetuning in the inner loop of meta-learning with prototypical networks

## Background

### Prototypical Networks and Meta-learning

A prototypical network is a form of meta-learning. Meta-learning is the process of learning an inductive bias for your model. This inductive bias can take a variety of forms. Often this inductive bias consists in how model weights are initialized before they are finetuned on a target task. They're initialized in such a way that finetuning happens more efficiently (requiring less data). *Meta-learning is therefore a training loop that operates on a model's initial weights.* 

Meta-learning usually involves an inner loop that actually finetunes the model. The outcome of the finetuning will be influenced by how the weights are initialized: it will take more or less training data to converge. Once finetuning is over, the outer loop takes stock of the outcome of this finetuning in order to update how it's going to initialize the weights. In the case of a strict prototypical network, the inner finetuning loop simply consists in a single step where we embed labelled examples ("support examples") using the encoder with its initial weights, and then we average those support vectors for each class. There is thus no backpropagation or weight update during this finetuning.

The outer loop does however update the model weights. It fetches a couple of labelled examples (called "queries"), embeds them, and computes their distance to each prototype. The loss used for backprop is the average negative log-likelihood (NLL) of each query's distance to the correct prototype. The gradient update is applied to the model's initial weights. Since the inner loop here doesn't update the weights, the initial weights are just the current weights. At each iteration of the outer loop, we learn new initial weights that will give the model a good inductive bias. In the case of a prototypical network, a good inductive bias means that the encoder's weights are such that the encoder will represent support examples and query examples so that query examples will be more likely to be close to the right support examples.

What kind of data should the outer loop feed into the inner loop? That depends on what the initialized model will be trained on at meta-test time. During meta-testing, we finetune and evaluate the model on a variety of tasks. For each task, we start with the model initialized by the meta-training procedure described above, finetune it with the training data available for that task, and evaluate it on the test data. (Again, in the case of a strict prototypical network, finetuning just means building prototypes out of the support examples.) Since the same initial model is used for multiple tasks at meta-test time, the inductive bias needs to be good for a variety of tasks. That's why it's important for the model to be exposed to a variety of tasks during meta-training. Each iteration of the outer loop should therefore pick a random task and feed that to the inner loop.

### Optimization-based meta-learning

We've emphasized that, in prototypical networks, the finetuning inner loop just consists in building prototypes out of support examples and that no backprop is involved. However, we *could* do some backprop in the inner loop. What would that look like? 

We've seen how queries are used at the bottom of the outer loop to calculate the NLL loss and update the initial model weights. Instead, we could do that as part of the inner loop. In the inner loop, we would use some examples as supports to build the prototypes, and then use the rest as queries to compute NLL of their distance to the correct prototype, backprop, and update the model weights. We would go through the queries in mini-batches, therefore having a proper multi-step inner loop instead of the single-step inner loop in strict prototypical networks. 

In the inner loop, we run through T batches of queries and update the model weights away from their initial value. This means that we need to save the initial model weights at the beginning of the inner loop, since that's what the outer loop needs to update at the end of the inner loop. The gradient for the outer loop, to be applied on the initial weights, is therefore a "meta-gradient". This meta-gradient corresponds to the gradient of the operation consisting in the inner loop's T-step optimization. This multi-step optimization can be abstracted away as a single operation (or "function") with an associated gradient.

This meta-gradient involves the second-order derivates of the NLL losses in the inner loop. It can therefore be computationally expensive. That said, it's been shown that the first-order approximation of that meta-gradient works pretty well. That approximation is simply the difference between the weight values at the end of the T-step inner-loop optimization and the weight values at the beginning of the inner loop. Keeping a copy of the weights at the beginning of the inner loop, and subtracting that from the weights at the end of the inner loop, is therefore all we need. This meta-gradient is then applied to that copy of the initial weights. Finally, we plug in those initial weights updated with the meta-gradient into the model when the next inner loop starts.

## Setup

* `source activate pytorch_p36`
* `cd <this repo>`
* `python setup.py install`
* `pip install -r requirements.txt`

## Data

* `python scripts/prepare_data.py --data-dir <data dir> --input <tsv file> --disjoint`
* Input tsv file should be in the classic Alexa 3-col format (domain/skill, intent, anno)
* Example tsv file can be found here: `/apollo/env/HoverboardDefaultMLPS3Tool/bin/mlps3 cp -k "com.amazon.snl-3pi18n.team.hoverboard" s3://hoverboard-shared-snl-3pi18n-eu-west-1/nlu/datasets/fud.tsv.tgz .`
* The input tsv can have a mix of 3P skills and 1P domains as tasks
* The resulting folder in the data dir is as follows:

```
meta/
    train.tsv
    valid.tsv
tasks/
    <task name 1>/
        train.tsv
        test.tsv
    <task name 2>/
        train.tsv
        test.tsv
```

* The `tasks` folder is used for meta-testing on a variety of tasks. A task can refer to a 1P domain or a 3P skill. For a 1P task, the training data consists in the goldens for that task (domain). For a 3P task, the training data consists in the interaction model of that task (skill). The tasks used in meta-testing can be disjoint (`--disjoint`) or not from those used during meta-training. Test data during meta-testing is annotated customer live traffic.

## Model

* `/apollo/env/HoverboardDefaultMLPS3Tool/bin/mlps3 cp -r s3://blu-core-model-training-eu/snl/models/huggingface ~/.cache`
* Updated config file: `emacs <yaml config file>`
* `python scripts/train_model.py -c <yaml config file>`

## Tests

### Pre-requisites

* In root directory, have the folder `data/meta` with a `train.tsv` file in it

### Run tests

* `pytest`