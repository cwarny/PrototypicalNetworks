import argparse
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from proto.data import (ClassDataset, TabularDataset, 
    TransformDataset, CategorizeProcessor, collate)
from proto.model import ProtoNet
from proto.util import load_weights, compose, dict_logger

torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-path', type=str, required=True)
    parser.add_argument('--saved-model', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

def evaluate_model(
        task_path, 
        saved_model, 
        device, 
        shots=5, 
        batch_size=32
    ):
    tp = Path(task_path)
    train_ds = ClassDataset.from_tsv(tp/'train.tsv', 
        names=['task','class','text'], n_support=shots)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tok = lambda x: tokenizer(x)['input_ids'] if len(x) else x
    cat = CategorizeProcessor()
    _ = cat(train_ds.df['class']) # learn mapping
    transforms = {
        'supports': tok,
        'queries': tok,
        'class': cat.proc1
    }
    train_ds = TransformDataset(train_ds, transforms)
    collate_fn = partial(collate, tokenizer.pad_token_id)
    train_dl = DataLoader(train_ds, batch_size=len(train_ds), 
        collate_fn=collate_fn)

    model = load_weights(ProtoNet(), saved_model).to(device)
    model.eval()
    # "train"
    for xb,(_,pos_to_id) in train_dl:
        xs,xq = xb
        n_class = xs.size(0)
        n_support = xs.size(1)
        xs = xs.view(n_class*n_support, -1)
        xs = xs.to(device)
        z_proto = model.encode(xs) \
            .view(n_class, n_support, -1) \
            .mean(1)
        # there will only be a single iteration
        # of this loop
    pos_to_id = pos_to_id.to(device)
    # test
    test_ds = TabularDataset.from_tsv(tp/'test.tsv', 
        names=['task','class','text'])
    def transform(d):
        x,y = d
        return {
            'supports': [],
            'queries': [tok(x)],
            'class': cat.proc1(y)
        }
    test_ds = TransformDataset(test_ds, transform)
    bs = batch_size
    test_dl = DataLoader(test_ds, batch_size=bs, 
        collate_fn=collate_fn)

    acc = 0
    c = 0
    for xb,yb in test_dl:
        c += bs
        xs,xq = xb
        xq = xq.to(device)
        zq = model.encode(xq.squeeze(1))
        out = model.compute_distances(zq, z_proto)
        y_hat = pos_to_id.gather(-1,out.argmax(-1))
        acc += torch.eq(y_hat, yb[1]).float().mean().item()*bs

    dict_logger({
        'task': tp.name,
        'n_class': len(train_ds),
        'accuracy': f"{acc/c:.6f}"
    })

if __name__ == '__main__':
    args = parse_args()
    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    evaluate_model(
        args.task_path,
        args.saved_model,
        device,
        shots=args.shots,
        batch_size=args.batch_size
    )