import yaml
import json
import functools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable
import torch

def compose(functions):
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)

def listify(x, max_depth=0, d=0):
    if x is None: return []
    if isinstance(x, list):
        if d < max_depth: return [listify(e, d=d+1) for e in x]
        return x
    if isinstance(x, str): return [x]
    if isinstance(x, Iterable): return list(x)
    return [x]

def parallel(func, arr, max_workers=4):
    if max_workers<2:
        results = list(map(func, enumerate(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(func, enumerate(arr)))
    if any([o is not None for o in results]):
        return results

def dict_logger(x):
    print(json.dumps(x, indent=4))

def safe_save_path(fp):
    fp = Path(fp)
    if not fp.parent.exists():
        fp.parent.mkdir(parents=True)

def save_weights(model, fp):
    safe_save_path(fp)
    torch.save(model.state_dict(), fp)

def load_weights(model, fp):
    model.load_state_dict(torch.load(fp))
    return model

def load_config(fp):
    with open(fp) as infile:
        config = yaml.safe_load(infile)
    return config

def save_config(config, fp):
    safe_save_path(fp)
    with open(fp, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)