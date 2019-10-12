import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import bcolz
import numpy as np
import pickle
import pandas as pd
from collections import Counter

# load train test data
intervened_test = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.intervened_test.csv', comment='#')
intervened_train = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.intervened_train.csv', comment='#')
not_intervened_test = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.not_intervened_test.csv', comment='#')
not_intervened_train = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.not_intervened_train.csv', comment='#')

# load GloVe data into dict
vectors = bcolz.open(f'../glove.6B/extracted/glove.6B.300d.dat')[:]
words = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_words.pkl', 'rb'))
word2idx = pickle.load(open(f'../glove.6B/extracted/glove.6B.300d_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

# some checking
def max_no_of_words(my_list):
    max_word = 0
    index = 0
    for i, sentence in enumerate(my_list):
        a = len(sentence.split(" "))
        if max_word < a:
            max_word = a
            index = i
    return max_word

print("intervened test max words: " + str(max_no_of_words(list(intervened_test.text))))
print("intervened train max words: " + str(max_no_of_words(list(intervened_train.text))))
print("not intervened test max words: " + str(max_no_of_words(list(not_intervened_test.text))))
print("not intervened train max words: " + str(max_no_of_words(list(not_intervened_train.text))))

# filter out data with legnth > 200
f_intervened_train = intervened_train[intervened_train.text.str.len() <= 200]
f_intervened_test = intervened_test[intervened_test.text.str.len() <= 200]
f_not_intervened_train = not_intervened_train[not_intervened_train.text.str.len() <= 200]
f_not_intervened_test = not_intervened_test[not_intervened_test.text.str.len() <= 200]
print("len of intervened_train: " + str(len(f_intervened_train)))
print("len of intervened_test: " + str(len(f_intervened_test)))
print("len of not_intervened_train: " + str(len(f_not_intervened_train)))
print("len of not_intervened_test: " + str(len(f_not_intervened_test)))
print("ratio of intervened to not intervened: " + str((1062 + 278) / (2396 + 592)))

test_data = pd.concat([f_intervened_test, f_not_intervened_test])
train_data = pd.concat([f_intervened_train, f_not_intervened_train])
print('Percentage of INTERVENED train data: ' + str(len(f_intervened_train)/(len(f_intervened_train)+len(f_intervened_test)) * 100) + '%')
print('Percentage of NOT INTERVENED train data: ' + str(len(f_not_intervened_train)/(len(f_not_intervened_train)+len(f_not_intervened_test)) * 100) + '%')
print('Percentage of train data: ' + str(len(train_data) / (len(train_data) + len(test_data)) * 100) + '%')

# extracting words from train
train_sentences = list(train_data.text)

words = Counter()
for i, sentence in enumerate(train_sentences):
    # Store the sentence as a list of words/tokens
    train_sentences[i] = []
    for word in sentence.split(" "):
        words.update([word.lower()])
        train_sentences[i].append(word)
    
    #print("Extracting labels: " + str(i+1) + " / " + str(len(train_sentences)))
print("Extracting words: DONE")

# remove that likely don't exist (remove words from the vocab that only appears once)
words = {k:v for k,v in words.items() if v > 1}
# sort the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)
# add padding and unknown to the vocab so that they are assigned to an index
words = ['<pad>', '<unk>'] + words
word2idx = {o:i for i, o in enumerate(words)}
idx2words = {i:o for i, o in enumerate(words)}

# extend the glove dictionary with <pad> and <unk>
emb_dim = 300
glove['<pad>'] = np.random.normal(scale=0.6, size=(emb_dim, ))
glove['<unk>'] = np.random.normal(scale=0.6, size=(emb_dim, ))

# Create weighta matrix for the extracted words
emb_dim = 300
weights_matrix = np.zeros((len(idx2words), emb_dim))
words_found = 0
unknown_words = 0
for i, word in idx2words.items():
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = glove['<unk>']
        glove[word] = weights_matrix[i]
        unknown_words += 1

print('Number of words found: ' + str(words_found))
print('Number of unknown words (most probably typos): ' + str(unknown_words))

unknown_index = word2idx['<unk>']

# map train sentence to index
for i, sentence in enumerate(train_sentences):
    train_sentences[i] = [word2idx[word] if word in word2idx else unknown_index for word in sentence]

# map test sentence to index
test_sentences = list(test_data.text)
for i, sentence in enumerate(test_sentences):
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else unknown_index for word in sentence.split(" ")]

def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, review in enumerate(sentences):
        if len(review) != 0:
            features[i, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 200

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)
print('Train sentences shape: ' + str(train_sentences.shape))
print('Test sentences shape: ' + str(test_sentences.shape))

assert train_sentences.shape[1] == 200 
assert test_sentences.shape[1] == 200


# extract train and test labels (0 if not intervened, 1 if intervened)
train_labels = np.array(list(train_data.intervened))
test_labels = np.array(list(test_data.intervened))
print('Train labels shape: ' + str(train_labels.shape))
print('Test labels shape: ' + str(test_labels.shape))

train_data = TensorDataset(torch.from_numpy(train_sentences).type('torch.FloatTensor'), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences).type('torch.FloatTensor'), torch.from_numpy(test_labels))

batch_size = 200

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device('cuda')
    print('GPU is available')
else:
    device = torch.device('cpu')
    print('GPU is not available, CPU used')

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#     emb_layer.load_state_dict({'weight': weights_matrix})
    emb_layer.weight = nn.Parameter(weights_matrix)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
    
class BaselineModel(nn.Module):
                                        
    def __init__(self, weights_matrix, output_size, hidden_dim, n_layers, drop_prob=0.5):
        super(BaselineModel, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        # Batch size * embedding dimension
        embeds = self.embedding(x)

        # lstm_out - seq_len, batch_size, nu_layers * num_dim 
        # hidden (batch_size * hidden_dim, batch_size * hidden_dim)
        # abhinav : rename hidden to (h_n, c_n) and use h_n
        lstm_out, hidden = self.lstm(embeds,hidden)

        # assert h_n.size(1) == hidden_dim
        # try:
        #    h_n.size(1) == hidden_dim 
        # except:
        #     raise AssertionError(f"hidden size is {h_n.size()}")

        # remove this 
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
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

# (hyper)params
output_size = 1
embedding_dim = emb_dim
hidden_dim = 300
n_layers = 1

# instantiate model
model = BaselineModel(torch.from_numpy(weights_matrix).type('torch.FloatTensor'), output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)

learning_rate = 0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 2
counter = 0
clip = 5
valid_loss_min = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)
    
    for inputs, labels in train_loader:
        counter += 1
        print(str(counter) + ' / ' + str(len(train_data)))
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))