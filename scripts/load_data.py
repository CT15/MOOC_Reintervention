import pandas as pd
import numpy as np
import bcolz
import pickle

def load_train_test_val_data(path_to_intervened_data, path_to_not_intervened_data, seed, take, lw=None, max_words=0):
    # load data
    intervened_data = pd.read_csv(path_to_intervened_data, comment='#')
    not_intervened_data = pd.read_csv(path_to_not_intervened_data, comment='#')

    if lw != None:
        lw.write_log(f'Intervened length: {len(intervened_data)}')
        lw.write_log(f'Not intervened length: {len(not_intervened_data)}')
        lw.write_log(f'Intervened / Not_intervened: {len(intervened_data) / len(not_intervened_data)}')

    # shuffle and take data
    intervened_data = intervened_data.sample(frac=take/100, replace=False, random_state=seed)
    not_intervened_data = not_intervened_data.sample(frac=take/100, replace=False, random_state=seed)

    # train test validation split
    intervened_train, intervened_val, intervened_test = np.split(intervened_data, [int(.8 * len(intervened_data)), int(.9 * len(intervened_data))])
    not_intervened_train, not_intervened_val, not_intervened_test = np.split(not_intervened_data, [int(.8 * len(not_intervened_data)), int(.9 * len(not_intervened_data))])

    train_data = pd.concat([intervened_train, not_intervened_train], ignore_index=True)
    val_data = pd.concat([intervened_val, not_intervened_val], ignore_index=True)
    test_data = pd.concat([intervened_test, not_intervened_test], ignore_index=True)

    if max_words > 0:
        # filter out data with length > MAX_NUM_OF_WORDS
        test_data = test_data[test_data.text.str.split(" ").str.len() <= max_words]
        val_data = val_data[val_data.text.str.split(" ").str.len() <= max_words]
        train_data = train_data[train_data.text.str.split(" ").str.len() <= max_words]

        if lw != None:
            lw.write_log(f'len of test_data (before batching): {len(test_data)}')
            lw.write_log(f'len of train_data (before batching): {len(train_data)}')
            lw.write_log(f'len of val_data (before batching): {len(val_data)}')

    # shuffle the train test val data using (seed + 123)
    train_data = train_data.sample(frac=1, replace=False, random_state=seed+123)
    val_data = val_data.sample(frac=1, replace=False, random_state=seed+123)
    test_data = test_data.sample(frac=1, replace=False, random_state=seed+123)

    intervened_ratio = len(intervened_data) / (len(intervened_data) + len(not_intervened_data))
    return train_data, test_data, val_data, intervened_ratio


# Paths should be parameters, but whatever
def load_glove():
    # load GloVe Data into dict
    emb_dim = 300
    vectors = bcolz.open(f'../glove.6B/extracted/glove.6B.300d.dat')[:]
    words = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove, emb_dim