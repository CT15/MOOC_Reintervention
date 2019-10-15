import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import bcolz
import numpy as np
import pickle
import pandas as pd
from collections import Counter
import argparse
import os
from datetime import datetime
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt

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

def write_log(text, _print=True):
    if os.path.exists(log_file_path):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    f = open(log_file_path, append_write)
    f.write(f'{text}\n')
    f.close()

    if _print:
        print(text)

assert take >= 1 and take <= 100

# Logging
write_log(f'baseline_train_v2 >> Run on {datetime.now()}')
write_log(f'Path to intervened data: {path_to_intervened_data}')
write_log(f'Path to not intervened data: {path_to_not_intervened_data}')
write_log(f'Path to not intervened data: {path_to_not_intervened_data}')
write_log(f'Seed: {seed}\nPercentage of data taken: {take}%\nEpochs specified: {epochs}')
write_log(f'Max number of words in a thread: {MAX_NUM_OF_WORDS}')
write_log(f'Batch size: {BATCH_SIZE}')

# load data
intervened_data = pd.read_csv(path_to_intervened_data, comment='#')
not_intervened_data = pd.read_csv(path_to_not_intervened_data, comment='#')

# shuffle and take data
intervened_data = intervened_data.sample(frac=take/100, replace=False, random_state=seed)
not_intervened_data = not_intervened_data.sample(frac=take/100, replace=False, random_state=seed)

# train test validation split
intervened_train, intervened_val, intervened_test = np.split(intervened_data, [int(.8 * len(intervened_data)), int(.9 * len(intervened_data))])
not_intervened_train, not_intervened_val, not_intervened_test = np.split(not_intervened_data, [int(.8 * len(not_intervened_data)), int(.9 * len(not_intervened_data))])

train_data = pd.concat([intervened_train, not_intervened_train], ignore_index=True)
val_data = pd.concat([intervened_val, not_intervened_val], ignore_index=True)
test_data = pd.concat([intervened_test, not_intervened_test], ignore_index=True)

# load GloVe Data into dict
emb_dim = 300
vectors = bcolz.open(f'../glove.6B/extracted/glove.6B.300d.dat')[:]
words = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_words.pkl', 'rb'))
word2idx = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# filter out data with length > MAX_NUM_OF_WORDS
test_data = test_data[test_data.text.str.split(" ").str.len() <= MAX_NUM_OF_WORDS]
val_data = val_data[val_data.text.str.split(" ").str.len() <= MAX_NUM_OF_WORDS]
train_data = train_data[train_data.text.str.split(" ").str.len() <= MAX_NUM_OF_WORDS]
write_log(f'len of test_data (before batching): {len(test_data)}')
write_log(f'len of train_data (before batching): {len(train_data)}')
write_log(f'len of val_data (before batching): {len(val_data)}')

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

# pad with index of <unk> at the end of each sentence
def pad_input(sentences, seq_len):
    unk_index = word2idx['<unk>']
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, sentence in enumerate(sentences):
        features[i, 0:len(sentence)] = np.array(sentence)
    return features

seq_len = MAX_NUM_OF_WORDS

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)
val_sentences= pad_input(val_sentences, seq_len)

assert train_sentences.shape[1] == MAX_NUM_OF_WORDS
assert test_sentences.shape[1] == MAX_NUM_OF_WORDS
assert val_sentences.shape[1] == MAX_NUM_OF_WORDS

train_labels = np.array(list(train_data.intervened))
test_labels = np.array(list(test_data.intervened))
val_labels = np.array(list(val_data.intervened))

train_data = TensorDataset(torch.from_numpy(train_sentences).type('torch.FloatTensor'), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences).type('torch.FloatTensor'), torch.from_numpy(test_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences).type('torch.FloatTensor'), torch.from_numpy(val_labels))

train_loader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)  
test_loader =  DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
val_loader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available, CPU used')

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.weight = nn.Parameter(weights_matrix)

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

#####  THE MODEL  #####
class BaselineModel(nn.Module):
    def __init__(self, weights_matrix, output_size=1, hidden_dim=300, n_layers=1, drop_prob=0.5):
        super(BaselineModel, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
        # batch_first: input and output tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        x = x.long() # to int64 as parameter to embedding
        embeds = self.embedding(x)
  
        #assert embeds.size() == torch.Size([seq_len, BATCH_SIZE, emb_dim])
        assert hidden[0].size() == torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # hidden_state
        assert hidden[1].size() ==  torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # cell_state

        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #assert lstm_out.size() == torch.Size([seq_len, BATCH_SIZE, 1 * self.hidden_dim])
        assert hidden[0].size() == torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # hidden_state
        assert hidden[1].size() ==  torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # cell_state

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # put into linear layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden
#######################

model = BaselineModel(torch.from_numpy(weights_matrix).type('torch.FloatTensor'))
model.to(device)

###############
learning_rate = 0.005
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
clip = 5
###############

valid_loss_min = np.Inf

def evaluate(model, data_loader):
    # eval_losses = []
    # num_correct = 0
    h = model.init_hidden(BATCH_SIZE)
    preds = []
    truths = []

    model.eval()
    for inputs, labels in data_loader:
        h = tuple([each.data for each in h])
        assert h[0].size() == torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # hidden_state
        assert h[1].size() ==  torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # cell_state

        inputs, labels = inputs.to(device), labels.to(device)
        output, h = model(inputs, h)

        # eval_loss = criterion(output.squeeze(), labels.float())
        # eval_losses.append(eval_loss.item())
        pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        # correct_tensor = pred.eq(labels.float().view_as(pred))
        # correct = np.squeeze(correct_tensor.cpu().numpy())
        # num_correct += np.sum(correct)
        preds.append(pred.tolist())
        truths.append(labels.squeeze().tolist())
    
    # accuracy = num_correct/len(data_loader.dataset)
    # average_losses = np.mean(eval_losses)

    preds = [int(pred) for predlist in preds for pred in predlist]
    truths = [truth for truthlist in truths for truth in truthlist]
    
    # return accuracy, average_losses, preds, truths
    return truths, preds

#####  TRAINING #####
write_log('##### TRAINING #####')

epoch_with_min_loss = 0
min_loss = np.Inf

# for plotting purpose
y_axis = []
x_axis = []
for epoch in range(1, epochs+1):
    model.train()

    h = model.init_hidden(BATCH_SIZE)
    batch_completed = 0 # for progress tracking

    for inputs, labels in train_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        # This line is so that the buffers will not be freed when trying to backward through the graph
        h = tuple([each.data for each in h])

        assert h[0].size() == torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # hidden_state
        assert h[1].size() ==  torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # cell_state

        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        batch_completed += 1
        # Feel like too many things are printed out...
        # write_log(f'Epoch: {epoch} | batch completed: {batch_completed} / {int(len(train_data)//BATCH_SIZE)} | Training Loss: {loss}')
    
    if min_loss > loss:
        epoch_with_min_loss = epoch
        min_loss = loss

    y_axis.append(loss)
    x_axis.append(epoch)

    # Test the f1 of training and validation. Break if training f1 >= validation f1 (both grater than 0)
    train_truths, train_preds = evaluate(model, train_loader)
    val_truths, val_preds = evaluate(model, val_loader)
    train_f1 = f1_score(train_truths, train_preds)
    val_f1 = f1_score(val_truths, val_preds)

    write_log(f'>> End of Epoch {epoch} | Train F1: {round(train_f1, 3)} | Validation F1: {round(val_f1, 3)}')

    if train_f1 > 0 and val_f1 > 0 and val_f1 - train_f1 >= 0:
        write_log(f'*** Break training at epoch {epoch} ***')
        break

truths, preds = evaluate(model, test_loader)
test_f1 = f1_score(truths, preds)
precision = precision_score(truths, preds)
recall = recall_score(truths, preds)

# Plot
plt.plot(x_axis, y_axis, 'ro')
plt.plot(x_axis, y_axis)
plt.savefig(f'{os.path.dirname(os.path.abspath(__file__))}/results/baseline.seed{seed}.take{take}.epoch{epochs}.png')
print('image saved')

write_log(f'Precision (test): {round(precision, 3)}')
write_log(f'Recall (test): {round(recall, 3)}')
write_log(f'F1 (test): {round(test_f1, 3)}')
write_log('-----')
write_log(f'Epoch with min loss: {epoch_with_min_loss} | Loss: {min_loss}')
write_log('Try to run the model using this epoch value.')
write_log('===== END =====')
