import argparse
from functools import partial
import torch
import torch.optim as optim
from transformers import BertTokenizer
from proto.data import DataBunch, read_tsv, load_data
from proto.train import (Learner, StopEarly, Measure,
    Schedule, SetDevice)
from proto.model import ProtoNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--train-path', type=str, default='train.tsv')
    parser.add_argument('--valid-path', type=str, default='valid.tsv')
    parser.add_argument('--n-way', type=int, default=60)
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--n-support', type=int, default=5)
    parser.add_argument('--n-query', type=int, default=5)
    parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--decay-every', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.)
    return parser.parse_args()

def loss(out, yb):
    '''
    `out` has shape (n_class, n_query, n_class)
    There are `n_query*n_class` total query vectors
    and therefore a total of `n_query*n_class*n_class` 
    (logsoftmaxed) distances: each query vector has 
    a distance to each `n_class` prototype, 
    and one of these distances is the distance to 
    the true class. Learning proceeds by minimizing 
    the logsoftmaxed distance to the true class.
    The distance to the true class of the i-th 
    query vector for the j-th class corresponds to 
    out[j,i,j], so we use the `gather` method
    to get us a tensor of shape (n_class, n_query)
    of the distances to the true class for each
    `n_query` vector in each class.
    '''
    loss = -out.gather(-1, yb).mean()
    return loss
    
def accuracy(out, yb):
    '''
    Get the "predicted class" by retrieving the
    minimum distance (max negative distance) for
    each query vector
    '''
    _, y_hat = out.max(-1)
    return torch.eq(y_hat, yb.squeeze()).float().mean()

def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    def load(path):
        df = read_tsv(path, names=['intent', 'text', 'ner'])
        cache = {
            k: g['text'].tolist() for k,g in df.groupby('intent')
            if len(g) >= args.n_support+args.n_query
        }
        class_names = list(cache.keys())
        return load_data(class_names, cache, tokenizer, args)
    
    data = DataBunch.from_data_dir(args.data_dir, 
        args.train_path, args.valid_path, load_func=load)

    callbacks = [
        Measure(accuracy),
        SetDevice(args.device),
        StopEarly(),
        Schedule(args.decay_every)
    ]

    model = ProtoNet(bert_model_name=args.bert_model_name)

    opt_func = partial(optim.Adam, 
        lr=args.learning_rate, weight_decay=args.weight_decay)
    
    learner = Learner(model, data, loss, opt_func, callbacks=callbacks)
    learner.fit(args.n_epochs)

if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)