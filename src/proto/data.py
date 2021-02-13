from functools import partial, reduce
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import logging
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
    def __init__(self, df, n_support=-1, n_query=0):
        super(ClassDataset, self).__init__()
        self.n_support = n_support
        self.n_query = n_query
        self.df = df
        original_group = list(df.groupby('class'))
        filtered_group = []
        for k, g in original_group:
            if len(g) < n_support + n_query:
                logging.warning(f'You requested '
                                f'{n_support} supports and {n_query} queries per class '
                                f'but class {k} only has {len(g)} samples.\n'
                                f'Class {k} is removed from the dataset.')
            else:
                filtered_group.append((k, g))
        self.grouped = filtered_group

    @classmethod
    def from_tsv(cls, fp, *args, names=None, converters=None, **kwargs):
        df = read_tsv(fp, names=names, converters=converters)
        return cls(df, *args, **kwargs)
    
    def __getitem__(self, i):
        k = self.grouped[i][0]
        if self.n_support == -1: # return all samples as supports
            samples = self.grouped[i][1]
            samples = samples['text'].tolist()
            return {
                'supports': samples,
                'queries': [],
                'class': k
            }
        n = self.n_support + self.n_query
        samples = self.grouped[i][1].sample(n)
        samples = samples['text'].tolist()
        return {
            'supports': samples[:self.n_support],
            'queries': samples[self.n_support:],
            'class': k
        }
    
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
        return (
            self.df.iloc[i]['text'],
            self.df.iloc[i]['class']
        )
    
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

class Processor:
    def __call__(self, items):
        return self.process(items)

    def proc1(self, item):
        return item
    
    def process(self, items): 
        return [self.proc1(item) for item in items]
    
    def deproc1(self, item):
        return item

    def deprocess(self, items):
        return [self.deproc1(item) for item in items]

class CategorizeProcessor(Processor):
    def __init__(self, *args, unk=None, ids_to_categories=None, 
            max_vocab=60000, min_freq=0, **kwargs):
        super(CategorizeProcessor, self).__init__(*args, **kwargs)
        unk = unk or ('<unk>',0)
        assert isinstance(unk,tuple) and len(unk)==2 \
            and isinstance(unk[0],str) and isinstance(unk[1],int)
        self.ids_to_categories = ids_to_categories
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.unk = unk
        self.special_categories = [unk]
    
    def __call__(self, items):
        if self.ids_to_categories is None:
            freq = Counter(items)
            self.ids_to_categories = [o for o,c in freq.most_common() 
                if c >= self.min_freq]
            for cat,idx in sorted(self.special_categories, 
                key=lambda d:d[1]):
                if cat in self.ids_to_categories: 
                    self.ids_to_categories.remove(cat)
                self.ids_to_categories.insert(idx, cat)
        if getattr(self, 'categories_to_ids', None) is None:
            self.categories_to_ids = defaultdict(lambda: self.unk[1], {
                v:k for k,v in enumerate(self.ids_to_categories)
            })
        return super(CategorizeProcessor, self).__call__(items)
    
    def proc1(self, category):
        return self.categories_to_ids[category]
    
    def deproc1(self, idx):
        return self.ids_to_categories[idx]

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
    xs, xq, ks = [],[],[]
    max_len = 0
    n_class = 0
    for s in samples:
        n_class += 1
        xs.append(s['supports'])
        xq.append(s['queries'])
        ks.append(s['class'])
        for d in s['supports']+s['queries']: 
            max_len = max(max_len, len(d))
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
    if n_query:
        idx = torch.arange(0,n_class) \
            .view(n_class,1,1) \
            .expand(n_class,n_query,1).long() \
            .contiguous() \
            .view(n_class*n_query,-1)
    else:
        idx = LongTensor([])
    yb = idx,LongTensor(ks)
    return xb,yb

def get_dl(fp, tokenizer, categorizer, n_way, n_episodes, pad_token_id=0, 
        n_support=-1, n_query=0):
    ds = ClassDataset.from_tsv(fp, 
        names=['task','class','text'], 
        n_support=n_support, n_query=n_query)
    _ = categorizer(ds.df['class']) # learn mapping
    transforms = {
        'supports': tokenizer,
        'queries': tokenizer,
        'class': categorizer.proc1
    }
    ds = TransformDataset(ds, transforms)
    sampler = TaskSampler(len(ds), n_way, n_episodes)
    collate_fn = partial(collate, pad_token_id)
    return DataLoader(ds, batch_sampler=sampler, 
        collate_fn=collate_fn)

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