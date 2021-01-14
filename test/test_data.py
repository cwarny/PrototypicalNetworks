import pytest
import argparse
from pathlib import Path
import torch
from transformers import BertTokenizer
from proto.data import load_data, read_tsv, DataBunch

args = argparse.Namespace(
    data_dir='data',
    train_path='train.tsv',
    valid_path='valid.tsv',
    n_way=3,
    n_episodes=4,
    n_support=5,
    n_query=6,
    bert_model_name='bert-base-uncased'
)

@pytest.fixture(scope="module")
def tokenizer_fixt():
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    return tokenizer

@pytest.fixture(scope="module")
def cache_fixt():
    df = read_tsv(Path(args.data_dir)/args.train_path, 
        names=['intent', 'text', 'ner'], nrows=1000)
    cache = {
        k: g['text'].tolist() for k,g in df.groupby('intent')
        if len(g) >= args.n_support+args.n_query
    }
    return cache

def test_load_data(cache_fixt, tokenizer_fixt):
    class_names = list(cache_fixt.keys())
    dl = load_data(class_names, cache_fixt, tokenizer_fixt, args)
    assert len(dl) == args.n_episodes
    for xb,yb in dl: break # load one batch
    # input batch has both support and query vectors
    assert isinstance(xb, tuple) and len(xb) == 2
    xs,xq = xb
    assert isinstance(xs, torch.Tensor)
    assert isinstance(xq, torch.Tensor)
    assert isinstance(yb, torch.Tensor)
    assert xs.size(0) == xq.size(0) == args.n_way, \
        'there should be {} classes per batch'.format(args.n_way)
    assert xs.size(1) == args.n_support, \
        'there should be {} support sequences'.format(args.n_support)
    assert xq.size(1) == args.n_query, \
        'there should be {} query sequences'.format(args.n_query)
    assert xs.size(2) == xq.size(2), \
        'support and query sequences should be padded to the same length'

def test_databunch_instantiation(cache_fixt, tokenizer_fixt):
    def load(path):
        class_names = list(cache_fixt.keys())
        return load_data(class_names, cache_fixt, tokenizer_fixt, args)
    data = DataBunch.from_data_dir(args.data_dir, 
        args.train_path, args.valid_path, load_func=load)
    assert hasattr(data, 'train_dl')
    assert hasattr(data, 'valid_dl')