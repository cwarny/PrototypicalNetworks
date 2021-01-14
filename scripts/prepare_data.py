import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from proto.data import read_tsv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--train', type=str, default='train.tsv')
    parser.add_argument('--valid', type=str, default='valid.tsv')
    parser.add_argument('--test', type=str, default='test.tsv')
    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    converters = {
        'intent': lambda row: '.'.join([row['domain'],row['intent']]),
        'text': lambda row: ' '.join([
            item.split('|')[0] 
            for item in row['annotation'].split(' ')
        ]),
        'ner': lambda row: ' '.join([
            item.split('|')[1] 
            for item in row['annotation'].split(' ')
        ])
    }
    df = read_tsv(data_dir/args.input, names=['domain', 'intent', 'annotation'], 
        converters=converters)
    train, test = train_test_split(df, train_size=.8, stratify=df.intent)
    train, valid = train_test_split(train, train_size=.8, stratify=train.intent)
    splits = {'train':train, 'valid':valid, 'test':test}
    for name,split in splits.items():
        path = getattr(args, name)
        split.to_csv(data_dir/path, sep='\t', index=False, 
            header=None, columns=['intent', 'text','ner'])

if __name__ == '__main__':
    args = parse_args()
    main(args)