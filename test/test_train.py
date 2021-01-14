import pytest
import math
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from proto.train import Learner, Callback, Measure, StopEarly, Meter
from proto.data import DataBunch

torch.manual_seed(0)

def test_learner():
    n_epochs = 2
    batch_size = 2
    data_size = 10

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.param = nn.Parameter(torch.zeros(batch_size))
        def forward(self, xb): 
            return xb*self.param
    
    class DummyDataset(Dataset):
        def __init__(self):
            super(DummyDataset, self).__init__()
            self.X = torch.zeros(data_size)
            self.y = torch.zeros(data_size)
        def __getitem__(self, i): return self.X[i], self.y[i]
        def __len__(self): return self.y.size(0)

    class DummyCallback(Callback):
        _order = 2
        def begin_fit(self):
            if hasattr(self,'log'): 
                self.learner.log.append('begin_fit')
            else: 
                self.learner.log = ['begin_fit']
        def begin_epoch(self):
            self.learner.log.append('begin_epoch')
            assert self.dl == self.data.train_dl
        def begin_batch(self): 
            self.learner.log.append('begin_batch')
            assert self.xb.size(0) == self.yb.size(0) == batch_size
        def after_forward(self): 
            self.learner.log.append('after_forward')
        def after_loss(self): 
            self.learner.log.append('after_loss')
        def after_backward(self): 
            self.learner.log.append('after_backward')
        def after_batch(self): 
            self.learner.log.append('after_batch')
        def begin_validate(self): 
            self.learner.log.append('begin_validate')
            assert self.dl == self.data.valid_dl
        def after_epoch(self): 
            self.learner.log.append('after_epoch')
        def after_cancel_train(self): 
            self.learner.log.append('after_cancel_train')
        def after_fit(self): 
            self.learner.log.append('after_fit')
            assert self.epoch < n_epochs, 'we should\'ve stopped early'

    train_dl = DataLoader(DummyDataset(), batch_size=batch_size)
    valid_dl = DataLoader(DummyDataset(), batch_size=batch_size)
    data = DataBunch(train_dl, valid_dl)

    def loss(out,yb): return out.mean()
    def acc(out,yb): return 0

    # ensure StopEarly runs after DummyCallback
    StopEarly._order = 3

    callbacks = [
        DummyCallback(),
        Measure(acc),
        StopEarly(patience=0)
    ]

    opt_func = partial(optim.Adam, lr=1e-3, weight_decay=0.)
    model = DummyModel()
    
    learner = Learner(model, data, loss, opt_func, 
        callbacks=callbacks)
    learner.fit(n_epochs)
    
    assert hasattr(learner, 'log')
    # the next assert checks that all the hooks in the 
    # dummy callback were correctly fired the right
    # number of times and in the right order
    assert learner.log == ['begin_fit', 'begin_epoch'] \
        + ['begin_batch','after_forward','after_loss','after_backward',
            'after_batch']*(data_size//batch_size) \
        + ['begin_validate'] \
        + ['begin_batch','after_forward','after_loss',
            'after_batch']*(data_size//batch_size) \
        + ['after_epoch','after_cancel_train','after_fit']
    assert callbacks[1].train_meter.in_train
    assert not callbacks[1].valid_meter.in_train
    assert callbacks[1].train_meter.avg_meters == [0,0]

def test_meter():
    m = Meter(lambda x: 0)
    assert isinstance(m.metrics, list)
    m.reset()
    assert m.tot_loss == 0
    assert m.count == 0
    assert len(m.tot_mets) == 1
    assert len(m.all_meters) == 2
    assert math.isnan(m.avg_meters[0])