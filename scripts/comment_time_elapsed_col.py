# This script appends `time_elapsed` column to comment
# tables. `time_elapsed` values are determined based on 
# `post_time`.

#! ../../venv/bin/python3

import pandas as pd
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path

VERSION = 1.0

parser = argparse.ArgumentParser(description='Append `time_elapsed` column to post / column table.')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to comment csv')


def comment_time_elapsed(df):
    df['time_elapsed'] = df.groupby('post_id')['post_time'].diff()
    return df


if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']

    df = pd.read_csv(path, comment='#')
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    df = comment_time_elapsed(df)

    output_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/{Path(path).stem}.comment_time_elapsed_col.csv'
    script_file_name = os.path.basename(__file__)
    with open(output_file_path, 'w+') as f:
        f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
        f.write(f'# Run with command `python {script_file_name} -p {path}`\n')

    df.to_csv(output_file_path, mode='a', index=False)