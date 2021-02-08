import pytest
import argparse
from pathlib import Path
import torch
from transformers import BertTokenizer
from proto.data import get_dl, DataBunch

data_dir = Path('data/meta')
n_way = 3
n_support = 5
n_query = 6
n_episodes = 4
encoder_name = 'bert-base-uncased'

@pytest.fixture(scope="module")
def tokenizer_fixt():
    tokenizer = BertTokenizer.from_pretrained(encoder_name)
    return tokenizer

@pytest.fixture(scope="module")
def data_fixt(tokenizer_fixt):
    tok = lambda x: tokenizer_fixt(x)['input_ids']
    pad_token_id = tokenizer_fixt.pad_token_id
    dl = get_dl(data_dir/'train.tsv', tok, n_way, 
        n_episodes, pad_token_id=pad_token_id, 
        n_support=n_support, n_query=n_query)
    return dl

def test_load_data(data_fixt):
    assert len(data_fixt) == n_episodes
    for xb,yb in data_fixt: break # load one batch
    # input batch has both support and query vectors
    assert isinstance(xb, tuple) and len(xb) == 2
    assert isinstance(yb, tuple) and len(yb) == 2
    xs,xq = xb
    idx,yb = yb
    assert isinstance(xs, torch.Tensor)
    assert isinstance(xq, torch.Tensor)
    assert isinstance(idx, torch.Tensor)
    assert isinstance(yb, torch.Tensor)
    assert xs.size(0) == xq.size(0), \
        'support and query seqs should have the same number of classes'
    assert xq.size(0) == n_way
    assert xs.size(1) == n_support
    assert xq.size(1) == n_query
    assert xs.size(2) == xq.size(2), \
        'support and query seqs should be padded to the same length'

def test_databunch_instantiation(tokenizer_fixt):
    tok = lambda x: tokenizer_fixt(x)['input_ids']
    pad_token_id = tokenizer_fixt.pad_token_id
    data = DataBunch.from_data_dir(data_dir, tok, n_way, 
        n_episodes, n_support=n_support, n_query=n_query, 
        pad_token_id=pad_token_id)
    assert hasattr(data, 'train_dl')
    assert hasattr(data, 'valid_dl')