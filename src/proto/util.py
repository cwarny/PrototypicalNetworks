import json
import functools
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

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

def to_dict(k, v):
    return { k: v }

def dict_logger(x):
    print(json.dumps(x, indent=4))