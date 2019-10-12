#! ../../venv/bin/python3

import bcolz
import numpy as np
import pickle
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Parse glove txt file')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to glove txt file')

curr_dir = os.path.dirname(os.path.abspath(__file__)) # directory in which this file is located

def execute(path, filename):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{curr_dir}/results/{filename}.dat', mode='w')

    count = 1
    with open(path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            
            print(f'{count} / 400001')
            count += 1
            
    # 400k words and 1 <unk>
    vectors = bcolz.carray(vectors[1:].reshape((400001, 300)), rootdir=f'{curr_dir}/results/{filename}.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{curr_dir}/results/{filename}_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{curr_dir}/results/{filename}_idx.pkl', 'wb'))

    # To create a dictionary of key = word and value = vector

    # vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
    # words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
    # word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))
    # glove = {w: vectors[word2idx[w]] for w in words}
    # print(glove['the'])

if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']
    filename = Path(path).stem

    execute(path, filename)

    print('DONE! Check results folder for the following files and folder:')
    print('1. ' + filename + '_words.pkl')
    print('2. ' + filename + '_idx.pkl')
    print('3. ' + filename + '.dat')
