import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from proto.data import read_tsv

np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--disjoint', 
        help=('ensure meta-test tasks are not seen '
            'during meta-traininig'), 
        action='store_true')
    parser.add_argument('--min-samples', type=int, default=30,
        help='min number of samples per class')
    parser.add_argument('--meta-split', type=float, default=.8)
    parser.add_argument('--split', type=float, default=.5)
    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)

    converters = {
        'class': lambda row: '.'.join([row['task'],row['class']]),
        'text': lambda row: ' '.join([
            item.split('|')[0] 
            for item in row['annotation'].split(' ')
        ]),
        'ner': lambda row: ' '.join([
            item.split('|')[1] 
            for item in row['annotation'].split(' ')
        ])
    }
    df = read_tsv(data_dir/args.input, 
            names=['task', 'class', 'annotation'], 
            converters=converters) \
        .groupby('class') \
        .filter(lambda g: len(g)>=args.min_samples)
    
    # Split tasks into meta-training, meta-validation
    # and meta-testing
    if args.disjoint:
        train_tasks, test_tasks = train_test_split(
            df['task'].unique(), train_size=args.meta_split)
        train_tasks, valid_tasks = train_test_split(
            train_tasks, train_size=args.meta_split)
        meta_train = df[df['task'].isin(train_tasks)]
        meta_valid = df[df['task'].isin(valid_tasks)]
        meta_test = df[df['task'].isin(test_tasks)]
    else:
        meta_train, meta_test = train_test_split(df, 
            train_size=args.meta_split, stratify=df['task'])
        meta_train, meta_valid = train_test_split(meta_train, 
            train_size=args.meta_split, stratify=meta_train['task'])

    # split meta-test data into train and test
    train, test = train_test_split(meta_test, 
        train_size=args.split, stratify=meta_test['class'])
    
    meta_dir = data_dir/'meta'
    tasks_dir = data_dir/'tasks'
    if not meta_dir.exists(): meta_dir.mkdir()
    if not tasks_dir.exists(): tasks_dir.mkdir()
    
    splits = {'train': meta_train, 'valid': meta_valid}
    for split_name,split in splits.items():
        path = meta_dir/'{}.tsv'.format(split_name)
        split.to_csv(path, sep='\t', index=False, 
            header=None, columns=['task','class','text'])
    
    splits = {'train': train, 'test': test}
    for split_name,split in splits.items():
        for k,g in split.groupby('task'):
            d = tasks_dir/k
            if not d.exists(): d.mkdir()
            g.to_csv(d/'{}.tsv'.format(split_name), 
                sep='\t', index=False, header=None,
                columns=['task','class','text'])

if __name__ == '__main__':
    args = parse_args()
    main(args)