import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

from write_log import LogWriter
from load_data import load_train_test_val_data, load_glove
from model_utils import pad_input, to_data_loader, get_device, evaluate_model_f1, WeightedBCELoss
from BaselineModel import BaselineModel

###############
MAX_NUM_OF_WORDS = 200
BATCH_SIZE = 9
###############

parser = argparse.ArgumentParser(description='Train baseline model')
parser.add_argument('-i', action='store', 
                    type=str, required=True, help='Path to intervened data csv')
parser.add_argument('-n', action='store', 
                    type=str, required=True, help='Path to not intervened data csv')
parser.add_argument('-s', action='store',
                    type=int, required=True, help='Seed')
parser.add_argument('-t', action='store',
                    type=int, required=True, help='Percentage of data to take (int [10, 100])')
parser.add_argument('-e', action='store',
                    type=int, required=True, help='Epoch')

args = vars(parser.parse_args())

path_to_intervened_data = args['i']
path_to_not_intervened_data = args['n']
seed = args['s']
take = args['t']
epochs = args['e']

log_file_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.txt'

lw = LogWriter(log_file_path)

assert take >= 1 and take <= 100

lw.write_log(f'baseline_train_v2 >> Run on {datetime.now()}')
lw.write_log(f'Path to intervened data: {path_to_intervened_data}')
lw.write_log(f'Path to not intervened data: {path_to_not_intervened_data}')
lw.write_log(f'Path to not intervened data: {path_to_not_intervened_data}')
lw.write_log(f'Seed: {seed}\nPercentage of data taken: {take}%\nEpochs specified: {epochs}')
lw.write_log(f'Max number of words in a thread: {MAX_NUM_OF_WORDS}')
lw.write_log(f'Batch size: {BATCH_SIZE}')

train_data, test_data, val_data, intervened_ratio = load_train_test_val_data(path_to_intervened_data, path_to_not_intervened_data, seed, take, lw, MAX_NUM_OF_WORDS)

glove, emb_dim = load_glove()

# take notes of words in train data
train_sentences = list(train_data.text)

for sentence in train_sentences:
    assert len(sentence.split(" ")) <= MAX_NUM_OF_WORDS

words = Counter()
for i, sentence in enumerate(train_sentences):
    train_sentences[i] = [] # to feed in word by word into LSTM
    for word in sentence.split(" "):
        words.update([word.lower()])
        train_sentences[i].append(word)

# remove words that appear only once (likely typo)
words = {k:v for k,v in words.items() if v > 1}
# sort words => most common words first
words = sorted(words, key=words.get, reverse=True)

# Add punctuation into GloVe dictionary
if '.' in words:
    glove['.'] == np.random.normal(scale=0.6, size=(emb_dim, ))
if '!' in words:
    glove['!'] == np.random.normal(scale=0.6, size=(emb_dim, ))
if '?' in words:
    glove['?'] == np.random.normal(scale=0.6, size=(emb_dim, ))

words = ['<pad>', '<unk>'] + words
word2idx = {o:i for i, o in enumerate(words)}
idx2words = {i:o for i, o in enumerate(words)}

assert word2idx['<pad>'] == 0 and idx2words[0] == '<pad>'
assert word2idx['<unk>'] == 1 and idx2words[1] == '<unk>'

glove['<pad>'] = np.zeros((emb_dim,))
glove['<unk>'] = np.random.normal(scale=0.6, size=(emb_dim, ))

# create weights matrix for the extracted words
########################################################
# Note: 
# weights_matrix is like a customized embedding
# specific to our dataset. Usually this is a scaled down
# version of pre-trained embedding + some words that
# are not present in the pre-trained embedding
########################################################
weights_matrix = np.zeros((len(idx2words), emb_dim))

existing_words = 0
new_words = 0
for i, word in idx2words.items():
    try: 
        weights_matrix[i] = glove[word]
        existing_words += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
        glove[word] = weights_matrix[i]
        new_words += 1

assert existing_words + new_words == len(words)

# map train, test and val sentences to arrays of indices
unknown_index = word2idx['<unk>']

for i, sentence in enumerate(train_sentences):
    train_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in sentence]

test_sentences = list(test_data.text)
for i, sentence in enumerate(test_sentences):
    word_list = sentence.split(" ")
    assert len(word_list) <= MAX_NUM_OF_WORDS
    test_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in word_list]

val_sentences = list(val_data.text)
for i, sentence in enumerate(val_sentences):
    word_list = sentence.split(" ")
    assert len(word_list) <= MAX_NUM_OF_WORDS
    val_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in word_list]

seq_len = MAX_NUM_OF_WORDS

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)
val_sentences= pad_input(val_sentences, seq_len)

assert train_sentences.shape[1] == MAX_NUM_OF_WORDS
assert test_sentences.shape[1] == MAX_NUM_OF_WORDS
assert val_sentences.shape[1] == MAX_NUM_OF_WORDS

train_loader = to_data_loader(train_sentences, list(train_data.intervened), BATCH_SIZE)
test_loader = to_data_loader(test_sentences, list(test_data.intervened), BATCH_SIZE)
val_loader = to_data_loader(val_sentences, list(val_data.intervened), BATCH_SIZE)

device = get_device()

# save the weights_matrix so can be loaded for testing purpose
if not os.path.exists('baseline_weights_matrix.txt'):
    np.savetxt('baseline_weights_matrix.txt', weights_matrix, fmt='%d')

model = BaselineModel(torch.from_numpy(weights_matrix).type('torch.FloatTensor'))
model.to(device)

###############
learning_rate = 0.005
criterion = WeightedBCELoss(zero_weight=intervened_ratio, one_weight=1-intervened_ratio)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
clip = 5
###############

lw.write_log(f'Weight for 0: {intervened_ratio} | Weight for 1: {1-intervened_ratio}')

#####  TRAINING #####
lw.write_log('##### TRAINING #####')

# for plotting purpose
y_axis_epoch = []
x_axis_epoch = []
y_axis = []
x_axis = []
y_axis_model = []
x_axis_model = []
y_axis_f1_train = []
y_axis_f1_val = []
x_axis_f1 = []

highest_f1 = -1
saved_state_dict = None
model_generated = 0

model.zero_grad() # can't zero gradient for every epoch because momentum (Adam)
model.train()
 
for epoch in range(1, epochs+1):
    training_loss = 0
    data_trained_so_far = 0
    batch_completed = 0 # for progress tracking

    h = model.init_hidden(BATCH_SIZE, device)

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        
        # This line is so that the buffers will not be freed when trying to backward through the graph
        h = tuple([each.data for each in h])

        assert h[0].size() == torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # hidden_state
        assert h[1].size() ==  torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # cell_state

        output, h = model(inputs, h)
        loss = criterion.loss(output.squeeze(), labels.float())

        training_loss += loss
        data_trained_so_far += BATCH_SIZE

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_completed += 1

        lw.write_log(f'Epoch: {epoch} | batch completed: {batch_completed} / {int(len(train_data)//BATCH_SIZE)} | Training Loss: {training_loss/data_trained_so_far}')

        if batch_completed % 100 == 0:
            val_f1, val_precision, val_recall = evaluate_model_f1(model, val_loader, BATCH_SIZE, device)
            train_f1, train_precision, train_recall = evaluate_model_f1(model, train_loader, BATCH_SIZE, device)
            lw.write_log(f'########## Batch Completed: {batch_completed} ##########')
            lw.write_log(f'Training F1: {train_f1} | Validation F1: {val_f1}')
            lw.write_log(f'Training precision: {train_precision} | Validation precision: {val_precision}')
            lw.write_log(f'Training recall: {train_recall} | Validation recall: {val_recall}')
            
            if val_f1 > highest_f1 + 0.02:
                model_generated += 1
                model_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.model{model_generated}.pt'
                torch.save(saved_state_dict, model_path)
                y_axis_model.append(training_loss / data_trained_so_far)
                x_axis_model.append(epoch - 1 + (batch_completed / (len(train_data) / BATCH_SIZE)))
                lw.write_log(f'Model {model_generated} generated!')
                highest_f1 = val_f1

            lw.write_log(f'########################################################')

            saved_state_dict = model.state_dict()
            y_axis_f1_train.append(train_f1)
            y_axis_f1_val.append(val_f1)
            x_axis_f1.append(epoch - 1 + (batch_completed / (len(train_data) / BATCH_SIZE)))

        y_axis.append(training_loss / data_trained_so_far)
        x_axis.append(epoch - 1 + (batch_completed / (len(train_data) / BATCH_SIZE)))

    val_f1, val_precision, val_recall = evaluate_model_f1(model, val_loader, BATCH_SIZE, device)
    train_f1, train_precision, train_recall = evaluate_model_f1(model, train_loader, BATCH_SIZE, device)
    lw.write_log(f'########## Batch Completed: {batch_completed} ##########')
    lw.write_log(f'Training F1: {train_f1} | Validation F1: {val_f1}')
    lw.write_log(f'Training precision: {train_precision} | Validation precision: {val_precision}')
    lw.write_log(f'Training recall: {train_recall} | Validation recall: {val_recall}')
    lw.write_log(f'########################################################')

    y_axis_epoch.append(training_loss / data_trained_so_far)
    x_axis_epoch.append(epoch)
    del y_axis[-1]
    del x_axis[-1]
    y_axis.append(training_loss / data_trained_so_far)
    x_axis.append(epoch)
    y_axis_f1_val.append(val_f1)
    y_axis_f1_train.append(train_f1)
    x_axis_f1.append(epoch)

lw.write_log(f'Model generated for the following x: {x_axis_model}')

# Generate model
model_generated += 1
model_path = f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.model{model_generated}.pt'
torch.save(model.state_dict(), model_path)
lw.write_log(f'Training done! Model {model_generated} generated!')

# Plot
plt.plot(x_axis_epoch, y_axis_epoch, 'ro')
plt.plot(x_axis, y_axis)
plt.plot(x_axis_model, y_axis_model, '^g')
plt.savefig(f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.png')

plt.clf()

plt.plot(x_axis_f1, y_axis_f1_train)
plt.plot(x_axis_f1, y_axis_f1_val, '-r')
plt.plot(x_axis_model, y_axis_model, '^g')
plt.savefig(f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.f1.png')

lw.write_log('##### TESTING #####')
test_f1, test_precision, test_recall = evaluate_model_f1(model, test_loader, BATCH_SIZE, device)
lw.write_log(f'Precision (test): {round(test_precision, 3)}')
lw.write_log(f'Recall (test): {round(test_recall, 3)}')
lw.write_log(f'F1 (test): {round(test_f1, 3)}')
lw.write_log('===== END =====\n')
