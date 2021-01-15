import random
from functools import partial
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader
from .util import compose, parallel, to_dict, listify

class ListDataset(Dataset):
    def __init__(self, lst, load=lambda x: x, path=None):
        super(ListDataset, self).__init__()
        self.lst = lst
        self.load = load

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, i):
        return self.load(self.lst[i])

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
    def __init__(self, chunksize=-1, max_workers=1):
        self.chunksize = chunksize
        self.max_workers = max_workers

    def __call__(self, items):
        if self.chunksize == -1:
            chunks = [items]
        else:
            chunks = [
                items[i:i+self.chunksize] 
                for i in range(0, len(items), self.chunksize)
            ]
        toks = parallel(self.proc_chunk, chunks, self.max_workers)
        # `toks` is a list of lists of lists of tokens
        # we want a list of lists of tokens instead
        # so here's a sneaky and efficient way to flatten 
        # a list of lists
        return list(itertools.chain.from_iterable(toks))
    
    def proc_chunk(self, args):
        _,chunk = args
        return self.process(chunk)

    def proc1(self, item):
        return item
    
    def process(self, items): 
        return [self.proc1(item) for item in items]

class TokenizeProcessor(Processor):
    def __init__(self, tokenizer, *args, **kwargs):
        super(TokenizeProcessor, self).__init__()
        self.tokenizer = tokenizer
    
    def proc1(self, text):
        return self.tokenizer(text)

class NumericalizeProcessor(Processor):
    def __init__(self, unk, pad, *args, ids_to_tokens=None, 
            max_vocab=60000, min_freq=2, **kwargs):
        super(NumericalizeProcessor, self).__init__(*args, **kwargs)
        assert isinstance(unk,tuple) and len(unk)==2 \
            and isinstance(unk[0],str) and isinstance(unk[1],int)
        assert isinstance(pad,tuple) and len(pad)==2 \
            and isinstance(pad[0],str) and isinstance(pad[1],int)
        self.ids_to_tokens = ids_to_tokens
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.unk, self.pad = unk, pad
        self.special_tokens = [unk, pad]
    
    def __call__(self, items):
        if self.ids_to_tokens is None:
            freq = Counter(tok for text in items for tok in text)
            self.ids_to_tokens = [
                tok for tok,c in freq.most_common(self.max_vocab) 
                if c >= self.min_freq
            ]
            for tok,idx in sorted(self.special_tokens, key=lambda d:d[1]):
                if tok in self.ids_to_tokens: self.ids_to_tokens.remove(tok)
                self.ids_to_tokens.insert(idx, tok)
        if getattr(self, 'tokens_to_ids', None) is None:
            self.tokens_to_ids = defaultdict(lambda: self.unk[1], {
                v:k for k,v in enumerate(self.ids_to_tokens)
            })
        return super(NumericalizeProcessor, self).__call__(items)
    
    def proc1(self, tokens):
        return [self.tokens_to_ids[tok] for tok in tokens]

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
            for cat,idx in sorted(self.special_categories, key=lambda d:d[1]):
                if cat in self.ids_to_categories: self.ids_to_categories.remove(cat)
                self.ids_to_categories.insert(idx, cat)
        if getattr(self, 'categories_to_ids', None) is None:
            self.categories_to_ids  = defaultdict(lambda: self.unk[1], {
                v:k for k,v in enumerate(self.ids_to_categories)
            })
        return super(CategorizeProcessor, self).__call__(items)
    
    def proc1(self, category):
        return self.categories_to_ids[category]

class EpisodicBatchSampler:
    '''
    This sampler generates "episodes" by sampling 
    `classification_cardinality` classes from the `n_classes` 
    possible classes. It will do that for `n_episodes` 
    "episodes".
    An epoch of training is therefore composed of `n_episodes`
    "training episodes".
    This is slightly different from the concept of an epoch 
    in typical model training. 
    Typically, an epoch just means going through your dataset 
    once. Here, because of the peculiarities of the prototypical
    network approach, we define an epoch as a series of 
    episodes where, for each episode, we randomly pick a
    `classification_cardinality` classes. 
    In each episode, the model will try to do a 
    `classification_cardinality`-way classification of a set of 
    unlabelled "query vectors" based on a set of labelled 
    "support vectors".
    '''
    def __init__(self, n_classes, classification_cardinality, n_episodes):
        self.n_classes = n_classes
        self.classification_cardinality = classification_cardinality
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.classification_cardinality]

def collate(pad_idx, samples):
    xs, xq = [],[]
    max_len = 0
    n_class = 0
    for ds,dq in samples:
        n_class += 1
        xs.append(ds)
        xq.append(dq)
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
    yb = torch.arange(0,n_class) \
        .view(n_class,1,1) \
        .expand(n_class,n_query,1).long()
    return xb,yb

def read_tsv(*args, converters=None, **kwargs):
    df = pd.read_csv(*args, **kwargs, sep='\t')
    if converters:
        assert isinstance(converters, dict)
        for k,converter in converters.items():
            df[k] = df.apply(converter, 1)
    return df

def load_data(classes, data_per_class, tokenizer, n_episodes, episode_config):
    assert isinstance(episode_config, dict)
    def load_utterance_data(x): return data_per_class[x]
    def extract_episode(x):
        examples = random.sample(x, episode_config['shots']+episode_config['queries'])
        return examples[:episode_config['shots']], examples[episode_config['shots']:]
    def tokenize(x):
        return map(lambda d: tokenizer(d)['input_ids'], x)
    transforms = compose([
        load_utterance_data,
        extract_episode,
        tokenize
    ])
    ds = TransformDataset(ListDataset(classes), transforms)
    sampler = EpisodicBatchSampler(len(ds), 
        episode_config['classification_cardinality'], n_episodes)
    return DataLoader(ds, batch_sampler=sampler, 
        collate_fn=partial(collate, tokenizer.pad_token_id))

class DataBunch:
    def __init__(self, train_dl, valid_dl):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
    
    @property
    def train_ds(self): return self.train_dl.dataset
    @property
    def valid_ds(self): return self.valid_dl.dataset
    
    @classmethod
    def from_data_dir(cls, data_dir, relative_train_path, 
            relative_valid_path, load_func=lambda x: x):
        data_dir = Path(data_dir)
        train_dl = load_func(data_dir/relative_train_path)
        valid_dl = load_func(data_dir/relative_valid_path)
        return cls(train_dl, valid_dl)