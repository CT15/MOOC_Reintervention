import numpy as np
import torch
import argparse
import os
from pathlib import Path
import pandas as pd

from BaselineModel import BaselineModel
from model_utils import evaluate_model_f1, to_data_loader, get_device
from write_log import LogWriter

BATCH_SIZE = 9

parser = argparse.ArgumentParser(description='Test baseline model')
parser.add_argument('-i', action='store', 
                    type=str, required=True, help='Path to intervened data csv')
parser.add_argument('-n', action='store', 
                    type=str, required=True, help='Path to not intervened data csv')
parser.add_argument('-s', action='store',
                    type=int, required=True, help='Seed')
parser.add_argument('-t', action='store',
                    type=int, required=True, help='Percentage of data to take (int [10, 100])')
parser.add_argument('-m', action='store', 
                    type=str, required=True, help='Path to test data csv')

args = vars(parser.parse_args())

model_path = args['m']

model_file_name = Path(model_path).stem
log_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/test_baseline_{model_file_name}.txt'

lw = LogWriter(log_file_path)

weights_matrix = np.loadtxt('baseline_weights_matrix.txt', dtype=float)

model = BaselineModel(torch.from_numpy(weights_matrix).type('torch.FloatTensor'))
model.load_state_dict(torch.load(path))

test_loader = to_data_loader(inputs, labels, BATCH_SIZE)
device = get_device()

test_f1, test_precision, test_recall = evaluate_model_f1(model, test_loader, BATCH_SIZE, device)
lw.write_log(f'Precision (test): {round(test_precision, 3)}')
lw.write_log(f'Recall (test): {round(test_recall, 3)}')
lw.write_log(f'F1 (test): {round(test_f1, 3)}')



