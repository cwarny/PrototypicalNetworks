import random
from functools import partial, reduce
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from .util import compose, parallel

def read_tsv(*args, converters=None, **kwargs):
    df = pd.read_csv(*args, **kwargs, sep='\t')
    if converters:
        assert isinstance(converters, dict)
        for k,converter in converters.items():
            df[k] = df.apply(converter, 1)
    return df

class ClassDataset(Dataset):
    def __init__(self, df, tokenizer, n_support=-1, n_query=0):
        super(ClassDataset, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.grouped = list(df.groupby('class'))
        for k,g in self.grouped:
            assert len(g) >= n_support+n_query, ('You requested '
            f'{n_support} supports and {n_query} queries per class '
            f'but class {k} only has {len(g)} samples.')
        self.tokenizer = tokenizer
    
    @classmethod
    def from_tsv(cls, fp, *args, names=None, converters=None, **kwargs):
        df = read_tsv(fp, names=names, converters=converters)
        return cls(df, *args, **kwargs)
    
    def __getitem__(self, i):
        if self.n_support == -1: # return all samples as supports
            samples = self.grouped[i][1]
            samples = samples['text'].tolist()
            tokenized = self.tokenizer(samples)
            return tokenized, [[]]
        n = self.n_support + self.n_query
        samples = self.grouped[i][1].sample(n)
        samples = samples['text'].tolist()
        tokenized = self.tokenizer(samples)
        supports = tokenized[:self.n_support]
        queries = tokenized[self.n_support:]
        # i is the index of the class
        return supports, queries, [i]*len(queries)
    
    def __len__(self):
        return len(self.grouped)

class TabularDataset(Dataset):
    def __init__(self, df):
        super(TabularDataset, self).__init__()
        self.df = df
    
    @classmethod
    def from_tsv(cls, *args, **kwargs):
        return cls(read_tsv(*args, **kwargs))
    
    def __getitem__(self, i):
        return self.df.loc[i]
    
    def __len__(self):
        return len(self.df)
    
class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        super(TransformDataset, self).__init__()
        assert isinstance(transforms, dict) or callable(transforms), \
            'expected a dict of transforms or a function'
        if isinstance(transforms, dict):
            for k, v in transforms.items():
                assert callable(v), str(k) + ' is not a function'
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]
        if isinstance(self.transforms, dict):
            for k, transform in self.transforms.items():
                sample[k] = transform(sample[k])
        else:
            sample = self.transforms(sample)
        return sample

class TaskSampler:
    def __init__(self, n_class, cardinality, n_episodes):
        self.n_class = n_class
        self.cardinality = cardinality
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_class)[:self.cardinality]

def collate(pad_idx, samples):
    xs, xq, ys = [],[],[]
    max_len = 0
    n_class = 0
    for ds,dq,y in samples:
        n_class += 1
        xs.append(ds)
        xq.append(dq)
        ys.append(y)
        for d in ds+dq: max_len = max(max_len, len(d))
    def pad(data):
        n_examples = len(data[0])
        padded = torch.zeros(n_class, n_examples, 
            max_len).long() + pad_idx
        for i,d in enumerate(data): 
            for j,x in enumerate(d):
                padded[i,j,:len(x)] = LongTensor(x)
        return padded
    xs,xq = pad(xs),pad(xq)
    xb = xs,xq
    n_query = xq.size(1)
    idx = torch.arange(0,n_class) \
        .view(n_class,1,1) \
        .expand(n_class,n_query,1).long() \
        .contiguous() \
        .view(n_class*n_query,-1)
    yb = idx,LongTensor(ys)
    return xb,yb

def get_dl(fp, tokenizer, n_way, n_episodes, pad_token_id=0, 
        n_support=-1, n_query=0):
    ds = ClassDataset.from_tsv(fp, tokenizer, names=['task','class','text'], 
        n_support=n_support, n_query=n_query)
    sampler = TaskSampler(len(ds), n_way, n_episodes)
    collate_fn = partial(collate, pad_token_id)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)

class DataBunch:
    def __init__(self, train_dl, valid_dl):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
    
    @property
    def train_ds(self): return self.train_dl.dataset
    @property
    def valid_ds(self): return self.valid_dl.dataset
    
    @classmethod
    def from_data_dir(cls, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        train_dl = get_dl(data_dir/'train.tsv', *args, **kwargs)
        valid_dl = get_dl(data_dir/'valid.tsv', *args, **kwargs)
        return cls(train_dl, valid_dl)