import pytest
import argparse
from pathlib import Path
import torch
from transformers import BertTokenizer
from proto.data import load_data, read_tsv, DataBunch

config = {
    'data': {
        'data_dir': 'data',
        'train_path': 'train.tsv',
        'valid_path': 'valid.tsv'
    },
    'train': {
        'episode': {
            'classification_cardinality': 3,
            'shots': 5,
            'queries': 6
        },
        'episodes': 4,
    },
    'model': {
        'encoder': {
            'model_name': 'bert-base-uncased'
        }
    }
}

@pytest.fixture(scope="module")
def tokenizer_fixt():
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder']['model_name'])
    return tokenizer

@pytest.fixture(scope="module")
def data_per_class_fixt():
    data_dir = Path(config['data']['data_dir'])
    train_path = config['data']['train_path']
    df = read_tsv(data_dir/train_path, 
        names=['intent', 'text', 'ner'], nrows=1000)
    shots = config['train']['episode']['shots']
    queries = config['train']['episode']['queries']
    data_per_class = {
        k: g['text'].tolist() for k,g in df.groupby('intent')
        if len(g) >= shots+queries
    }
    return data_per_class

def test_load_data(data_per_class_fixt, tokenizer_fixt):
    classes = list(data_per_class_fixt.keys())
    episodes = config['train']['episodes']
    episode_config = config['train']['episode']
    dl = load_data(classes, data_per_class_fixt, tokenizer_fixt, 
        episodes, episode_config)
    assert len(dl) == episodes
    for xb,yb in dl: break # load one batch
    # input batch has both support and query vectors
    assert isinstance(xb, tuple) and len(xb) == 2
    xs,xq = xb
    assert isinstance(xs, torch.Tensor)
    assert isinstance(xq, torch.Tensor)
    assert isinstance(yb, torch.Tensor)
    assert xs.size(0) == xq.size(0), \
        'support and query seqs should have the same number of classes'
    assert xq.size(0) == episode_config['classification_cardinality']
    assert xs.size(1) == episode_config['shots']
    assert xq.size(1) == episode_config['queries']
    assert xs.size(2) == xq.size(2), \
        'support and query seqs should be padded to the same length'

def test_databunch_instantiation(data_per_class_fixt, tokenizer_fixt):
    episodes = config['train']['episodes']
    episode_config = config['train']['episode']
    def load(path):
        classes = list(data_per_class_fixt.keys())
        return load_data(classes, data_per_class_fixt, 
            tokenizer_fixt, episodes, episode_config)
    data_dir = Path(config['data']['data_dir'])
    train_path = config['data']['train_path']
    valid_path = config['data']['valid_path']
    data = DataBunch.from_data_dir(data_dir, train_path, 
        valid_path, load_func=load)
    assert hasattr(data, 'train_dl')
    assert hasattr(data, 'valid_dl')