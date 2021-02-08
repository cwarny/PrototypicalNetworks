import pytest
import torch
import torch.nn as nn
from proto.model import ProtoNet

def test_model():
    model = ProtoNet()
    assert hasattr(model, 'encoder')
    assert isinstance(model, nn.Module)
    vocab_size, max_len = 10, 5
    n_class, n_support, n_query = 2, 3, 4
    xs = torch.randint(vocab_size, (n_class, n_support, max_len))
    xq = torch.randint(vocab_size, (n_class, n_query, max_len))
    xb = (xs,xq)
    out = model(xb)
    assert len(out.shape) == 2
    assert out.size(0) == n_class*n_query
    assert out.size(1) == n_class