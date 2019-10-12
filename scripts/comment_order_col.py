# This script appends `order` column to comment
# tables. `order` values are determined based on 
# `post_time`.

#! ../../venv/bin/python3

import pandas as pd
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path

VERSION = 1.0

parser = argparse.ArgumentParser(description='Append `order` column to comment table.')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path of comment csv file')


def comment_order(df):
    df = df.sort_values(['post_id', 'post_time'])
    df["order"] = df.groupby("post_id").cumcount() + 1
    return df


if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']
    
    df = pd.read_csv(path, comment='#')
    df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)    
    df = comment_order(df)

    output_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/{Path(path).stem}.comment_order_col.csv'
    script_file_name = os.path.basename(__file__)
    with open(output_file_path, 'w+') as f:
        f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
        f.write(f'# Run with command `python {script_file_name} -p {path}`\n')

    df.to_csv(output_file_path, mode='a', index=False)