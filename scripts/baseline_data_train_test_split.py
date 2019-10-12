# Takes in data of intervened and not intervened thread.
# Separates it into intervened_train, not_intervened_train,
# intervened_test, not_intervened_test

#! ../venv/bin/python3

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import argparse

VERSION = 1.0

parser = argparse.ArgumentParser(description='Split data to intervened train/test and not intervened train/test')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to data csv')

def execute(path, filename):
    data = pd.read_csv(path, comment='#')
    intervened = data[data.intervened == 1]
    not_intervened = data[data.intervened == 0]

    # train data
    intervened_train = intervened.sample(n=int(len(intervened)*0.8), replace=False, random_state=293487)
    not_intervened_train = not_intervened.sample(n=int(len(not_intervened)*0.8), replace=False, random_state=903984)

    # test data
    intervened_train_index = set(intervened_train.index)
    not_intervened_train_index = set(not_intervened_train.index)
    intervened_test = intervened[~intervened.index.isin(intervened_train_index)]
    not_intervened_test = not_intervened[~not_intervened.index.isin(not_intervened_train_index)]

    for csv_data_type in ['intervened_train', 'intervened_test', 'not_intervened_train', 'not_intervened_test']:
        script_file_name = os.path.basename(__file__)
        output_file_path = f'{os.getcwd()}/results/{filename}.{csv_data_type}.csv'
        with open(output_file_path, 'w+') as f:
            f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
            f.write(f'# Run with command `python {script_file_name} -p {path}`\n')

        if csv_data_type == 'intervened_train':
            intervened_train.to_csv(output_file_path, mode='a', index=False) 

        if csv_data_type == 'intervened_test':
            intervened_test.to_csv(output_file_path, mode='a', index=False) 

        if csv_data_type == 'not_intervened_train':
            not_intervened_train.to_csv(output_file_path, mode='a', index=False)

        if csv_data_type == 'not_intervened_test':
            not_intervened_test.to_csv(output_file_path, mode='a', index=False)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']
    filename = Path(path).stem
    
    execute(path, filename)