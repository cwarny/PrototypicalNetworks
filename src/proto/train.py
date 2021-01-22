import torch
import torch.optim.lr_scheduler as lr_scheduler
from proto.util import listify, dict_logger

class CancelTrainException(Exception):
    pass

class CancelEpochException(Exception):
    pass

class CancelBatchException(Exception):
    pass

class Callback:
    _order = 0
    def set_learner(self, learner):
        self.learner = learner

    def __getattr__(self, k):
        return getattr(self.learner, k)
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        # if this callback has not implemented this "hook"
        # then return false ("hook" here refers to something
        # like "begin_fit", etc.)
        return False

class SetDevice(Callback):
    def __init__(self, device):
        self.device = device
        super(SetDevice, self).__init__()
    
    def begin_fit(self):
        self.model.to(self.device)
    
    def begin_batch(self):
        for name,t in [('xb',self.xb), ('yb',self.yb)]:
            if isinstance(t, dict):
                setattr(self.learner, name, {
                    k:v.to(self.device) 
                    for k,v in t.items()
                })
            elif isinstance(t, list):
                setattr(self.learner, name, [
                    x.to(self.device) 
                    for x in t
                ])
            elif isinstance(t, tuple):
                setattr(self.learner, name, tuple([
                    x.to(self.device) 
                    for x in t
                ]))
            else:
                setattr(self.learner, name, t.to(self.device))

class TrainEval(Callback):    
    def begin_epoch(self):
        self.model.train()
        self.learner.in_train = True
    
    def begin_validate(self):
        self.model.eval()
        self.learner.dl = self.data.valid_dl
        self.learner.in_train = False

class Meter:
    def __init__(self, metrics=None, in_train=True):
        self.metrics = listify(metrics)
        self.in_train = in_train
    
    def reset(self):
        self.tot_loss, self.count = 0, 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_meters(self):
        return [self.tot_loss] + self.tot_mets
    
    @property
    def avg_meters(self):
        if self.count: return [o/self.count for o in self.all_meters]
        else: return [float('nan') for o in self.all_meters]

    def accumulate(self, learner):
        batch_size = learner.yb.numel()
        self.tot_loss += learner.loss * batch_size
        self.count += batch_size
        for i,metric in enumerate(self.metrics):
            self.tot_mets[i] += metric(learner.out, learner.yb)*batch_size

class StopEarly(Callback):
    def __init__(self, patience=5, save_model_path=None):
        super(StopEarly, self).__init__()
        self.valid_meter = Meter()
        self.avg_loss_prev, self.avg_loss_curr = None, None
        self.patience, self.frustration = patience, 0
        self.avg_loss_best = float('inf')
        self.save_model_path = save_model_path
    
    def begin_epoch(self):
        self.valid_meter.reset()
    
    def after_loss(self):
        if not self.in_train:
            with torch.no_grad(): 
                self.valid_meter.accumulate(self.learner)
    
    def after_epoch(self):
        self.avg_loss_prev = self.avg_loss_curr
        self.avg_loss_curr = self.valid_meter.avg_meters[0]
        if self.avg_loss_prev is not None:
            if self.avg_loss_curr <= self.avg_loss_prev: 
                # loss got better
                self.frustration = 0
                if self.avg_loss_curr < self.avg_loss_best:
                    self.avg_loss_best = self.avg_loss_curr
                    if self.save_model_path:
                        torch.save(self.model.state_dict(), self.save_model_path)
            else: 
                # loss got worse
                self.frustration += 1
        if self.frustration >= self.patience:
            raise CancelTrainException()

class Measure(Callback):
    def __init__(self, metrics, logger=None):
        super(Measure, self).__init__()
        self.train_meter = Meter(metrics)
        self.valid_meter = Meter(metrics, in_train=False)
        self.logger = logger or dict_logger
    
    def begin_epoch(self):
        self.train_meter.reset()
        self.valid_meter.reset()
    
    def after_loss(self):
        meter = self.train_meter if self.in_train else self.valid_meter
        with torch.no_grad(): meter.accumulate(self.learner)
    
    def after_epoch(self):
        stats = {'epoch': self.epoch, 'train':{}, 'valid':{}}
        for m in [self.train_meter, self.valid_meter]:
            split = 'train' if m.in_train else 'valid'
            for metric,value in zip(['loss']+m.metrics, m.avg_meters):
                stats[split][metric.__name__] = f"{value:.6f}"
        self.logger(stats)

class Schedule(Callback):
    def __init__(self, decay_every):
        super(Schedule, self).__init__()
        self.decay_every = decay_every

    def begin_fit(self):
        self.learner.scheduler = lr_scheduler.StepLR(self.opt, 
            self.decay_every, gamma=0.5)

    def after_step(self):
        self.scheduler.step()

class Learner:
    ALL_CBS = {
        'begin_fit',
        'begin_epoch',
        'begin_batch',
        'after_forward',
        'after_loss',
        'after_backward',
        'after_step',
        'after_batch',
        'begin_validate',
        'after_epoch',
        'after_fit',
        'after_cancel_batch',
        'after_cancel_epoch',
        'after_cancel_train'
    }

    def __init__(self, model, data, loss_func, opt_func, callbacks=None):
        '''
        `data` is an object that should have the following
        two properties: `train_dl` and `valid_dl`, and both
        should be data loaders
        '''
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.opt_func, self.opt = opt_func, None
        self.in_train = False
        self.callbacks = [TrainEval()] + (callbacks or [])
    
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.out = self.model(self.xb)
            self('after_forward')
            self.loss = self.loss_func(self.out, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally:
            self('after_batch')
    
    def all_batches(self):
        try:
            for i, (xb,yb) in enumerate(self.dl):
                self.one_batch(i,xb,yb)
        except CancelEpochException:
            self('after_cancel_epoch')
    
    def do_begin_fit(self, epochs):
        self.epochs = epochs
        for cb in self.callbacks: cb.set_learner(self)
        self('begin_fit')
    
    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, reset_opt=False):
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.model.parameters())
        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch)
                self.all_batches()
                with torch.no_grad():
                    self('begin_validate') 
                    self.all_batches()
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
    
    def __call__(self, cb_name):
        assert cb_name in self.ALL_CBS
        # Go through all callbacks in order
        for cb in sorted(self.callbacks, key=lambda x: x._order):
            cb(cb_name)