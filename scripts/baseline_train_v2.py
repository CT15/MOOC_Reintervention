import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import bcolz
import numpy as np
import pickle
import pandas as pd
from collections import Counter

# load train test data
test_data = pd.read_csv('../data/baseline/single_intervention/baseline.single_intervention.test.shuffle_seed1_take1845.csv', comment='#')
train_data = pd.read_csv('../data/baseline/single_intervention/baseline.single_intervention.train.shuffle_seed1_take7329.csv', comment='#')

# load GloVe Data into dict
emb_dim = 300
vectors = bcolz.open(f'../glove.6B/extracted/glove.6B.300d.dat')[:]
words = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_words.pkl', 'rb'))
word2idx = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# filter out data with length > 200
test_data = test_data[test_data.text.str.split(" ").str.len() <= 200]
train_data = train_data[train_data.text.str.split(" ").str.len() <= 200]
# need these to be printed AGAIN at the end
a = len(test_data)
b = len(train_data)
c = len(test_data)/len(train_data)
print(f'len of test_data: {a}')
print(f'len of train_data: {b}')
print(f'ratio of test to train: {c}')

# take notes of words in train data
train_sentences = list(train_data.text)

for sentence in train_sentences:
    assert len(sentence.split(" ")) <= 200

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

# map train and test sentences to arrays of indices
unknown_index = word2idx['<unk>']

for i, sentence in enumerate(train_sentences):
    train_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in sentence]

test_sentences = list(test_data.text)
for i, sentence in enumerate(test_sentences):
    word_list = sentence.split(" ")
    assert len(word_list) <= 200
    test_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in word_list]

# pad with index of <unk> at the end of each sentence
def pad_input(sentences, seq_len):
    unk_index = word2idx['<unk>']
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, sentence in enumerate(sentences):
        features[i, 0:len(sentence)] = np.array(sentence)
    return features

seq_len = 200

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

assert train_sentences.shape[1] == 200
assert test_sentences.shape[1] == 200

train_labels = np.array(list(train_data.intervened))
test_labels = np.array(list(test_data.intervened))

assert train_labels.shape[0] == train_sentences.shape[0]
assert test_labels.shape[0] == test_labels.shape[0]

train_data = TensorDataset(torch.from_numpy(train_sentences).type('torch.FloatTensor'), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences).type('torch.FloatTensor'), torch.from_numpy(test_labels))

###############
BATCH_SIZE = 9
###############

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True) # how to not drop last? 
test_loader =  DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True) # how to not drop last?

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
epochs = 1
clip = 5
###############

valid_loss_min = np.Inf
completed = 0 # for progress tracking

#####  TRAINING #####
model.train()
for i in range(epochs):
    h = model.init_hidden(BATCH_SIZE)

    for inputs, labels in train_loader:
        # diff = BATCH_SIZE - inputs.size[0]
        # if diff > 0: # need to have dummy data to have a full batch
        #     dummy_sentences = torch.from_numpy(np.zeros((diff, seq_len))).type('torch.FloatTensor')
        #     dummy_labels = torch.from_numpy(np.zeros((diff, seq_len)))
        #     torch.cat((inputs, dummy_sentences), 0)
        #     torch.cat((labels, dummy_labels), 0)

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

        completed += 1
        print(f'{completed} / {int(len(train_data)//BATCH_SIZE)}')


test_losses = []
num_correct = 0
h = model.init_hidden(BATCH_SIZE)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    assert h[0].size() == torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # hidden_state
    assert h[1].size() ==  torch.Size([model.n_layers * 1, BATCH_SIZE, model.hidden_dim]) # cell_state

    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)

    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print(f'len of test_data: {a}')
print(f'len of train_data: {b}')
print(f'ratio of test to train: {c}')
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))