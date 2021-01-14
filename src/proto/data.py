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
    def __init__(self, elem_list, load=lambda x: x, path=None):
        super(ListDataset, self).__init__()
        self.list = elem_list
        self.load = load

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.load(self.list[i])

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
    def __init__(self, unk, pad, *args, vocab=None, 
            max_vocab=60000, min_freq=2, **kwargs):
        super(NumericalizeProcessor, self).__init__(*args, **kwargs)
        assert isinstance(unk,tuple) and len(unk)==2 \
            and isinstance(unk[0],str) and isinstance(unk[1],int)
        assert isinstance(pad,tuple) and len(pad)==2 \
            and isinstance(pad[0],str) and isinstance(pad[1],int)
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.unk, self.pad = unk, pad
        self.special_tokens = [unk, pad]
    
    def __call__(self, items):
        if self.vocab is None:
            freq = Counter(tok for text in items for tok in text)
            self.vocab = [
                o for o,c in freq.most_common(self.max_vocab) 
                if c >= self.min_freq
            ]
            for tok,idx in sorted(self.special_tokens, key=lambda d:d[1]):
                if tok in self.vocab: self.vocab.remove(tok)
                self.vocab.insert(idx, tok)
        if getattr(self, 'otoi', None) is None:
            self.otoi = defaultdict(lambda: self.unk[1], {
                v:k for k,v in enumerate(self.vocab)
            })
        return super(NumericalizeProcessor, self).__call__(items)
    
    def proc1(self, tokens):
        return [self.otoi[tok] for tok in tokens]

class CategorizeProcessor(Processor):
    def __init__(self, *args, unk=None, vocab=None, 
            max_vocab=60000, min_freq=0, **kwargs):
        super(CategorizeProcessor, self).__init__(*args, **kwargs)
        unk = unk or ('<unk>',0)
        assert isinstance(unk,tuple) and len(unk)==2 \
            and isinstance(unk[0],str) and isinstance(unk[1],int)
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.unk = unk
        self.special_categories = [unk]
    
    def __call__(self, items):
        if self.vocab is None:
            freq = Counter(items)
            self.vocab = [o for o,c in freq.most_common() 
                if c >= self.min_freq]
            for cat,idx in sorted(self.special_categories, key=lambda d:d[1]):
                if cat in self.vocab: self.vocab.remove(cat)
                self.vocab.insert(idx, cat)
        if getattr(self, 'otoi', None) is None:
            self.otoi  = defaultdict(lambda: self.unk[1], {
                v:k for k,v in enumerate(self.vocab)
            })
        return super(CategorizeProcessor, self).__call__(items)
    
    def proc1(self, category):
        return self.otoi[category]

class EpisodicBatchSampler:
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

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

def load_data(class_names, cache, tokenizer, args):
    def load_utterance_data(x): return cache[x]
    def extract_episode(x):
        examples = random.sample(x, args.n_support+args.n_query)
        return examples[:args.n_support], examples[args.n_support:]
    def tokenize(x):
        return map(lambda d: tokenizer(d)['input_ids'], x)
    transforms = compose([
        load_utterance_data,
        extract_episode,
        tokenize
    ])
    ds = TransformDataset(ListDataset(class_names), transforms)
    sampler = EpisodicBatchSampler(len(ds), args.n_way, 
        args.n_episodes)
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