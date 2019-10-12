# This script combines post and comment tables.
# The resulting csv will be saved as
# <name_of_post_csv>+<name_of_comment_csv>_combined.csv

#! ../../venv/bin/python3

import pandas as pd
import numpy as np
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path

VERSION = 1.0

parser = argparse.ArgumentParser(description='Combine post and comment tables.')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to post csv')
parser.add_argument('-c', action='store', 
                    type=str, required=True, help='Path to comment csv')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    post = args['p']
    comment = args['c']

    post_df = pd.read_csv(post, comment='#')
    comment_df = pd.read_csv(comment, comment='#')

    post_df.drop(post_df.columns[post_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    comment_df.drop(comment_df.columns[comment_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

    post_df.drop(['original'], axis=1, inplace=True)

    post_df['parent_id'] = np.full(len(post_df,), None)
    comment_df.rename(columns={'comment_text':'post_text', 'post_id':'parent_id'}, inplace=True)

    combined_df = pd.concat([post_df, comment_df], axis=0, sort=True, ignore_index=True)

    print('Number of rows = ' + str(len(combined_df)))
    

    output_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/{Path(post).stem}+{Path(comment).stem}.combine_post_comment.csv'
    script_file_name = os.path.basename(__file__)
    with open(output_file_path, 'w+') as f:
        f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
        f.write(f'# Run with command `python {script_file_name} -p {post} -c {comment}`\n')

    combined_df.to_csv(output_file_path, mode='a', index=False)