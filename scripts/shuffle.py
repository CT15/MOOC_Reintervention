import os
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime

VERSION = 1.0

parser = argparse.ArgumentParser(description='Sample csv and save to another csv')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to csv')
parser.add_argument('-n', action='store', 
                    type=int, required=True, help='Number of rows to sample')
parser.add_argument('-s', action='store', 
                    type=int, required=True, help='Seed')

def shuffle_csv(df, n, seed):
    df = df.sample(n=n, replace=False, random_state=seed)
    return df

if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']
    n = args['n']
    seed = args['s']

    df = pd.read_csv(path, comment='#')
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    df = shuffle_csv(df, n, seed)

    output_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/{Path(path).stem}.shuffle_seed{seed}_take{n}.csv'
    script_file_name = os.path.basename(__file__)
    with open(output_file_path, 'w+') as f:
        f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
        f.write(f'# Run with command `python {script_file_name} -p {path} -n {n} -s {seed}`\n')

    df.to_csv(output_file_path, mode='a', index=False)
