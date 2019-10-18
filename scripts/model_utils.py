import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    emb_layer.weight = nn.Parameter(weights_matrix)

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


# pad with index of <unk> at the end of each sentence
# index of <unk> must be zero!
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for i, sentence in enumerate(sentences):
        features[i, 0:len(sentence)] = np.array(sentence)
    return features


def to_data_loader(inputs, labels, batch_size, shuffle=False):
    labels = np.array(labels)
    inputs = np.array(inputs)
    data = TensorDataset(torch.from_numpy(inputs).type('torch.FloatTensor'), torch.from_numpy(labels))
    return DataLoader(data, shuffle=shuffle, batch_size=batch_size, drop_last=True)


# Either CPU or GPU
def get_device(_print=True):
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('GPU is not available, CPU used')

    return device


def evaluate_model_f1(model, data_loader, batch_size, device):
    # eval_losses = []
    # num_correct = 0
    h = model.init_hidden(batch_size, device)
    preds = []
    truths = []

    model.eval()
    for inputs, labels in data_loader:
        h = tuple([each.data for each in h])
        # model.n_layers * 1 (1 for unidirectional and 2 for bidirectional lstm)
        # assert h[0].size() == torch.Size([model.n_layers * 1, batch_size, model.hidden_dim]) # hidden_state
        # assert h[1].size() ==  torch.Size([model.n_layers * 1, batch_size, model.hidden_dim]) # cell_state

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
    
    model.train()
    
    f1 = f1_score(truths, preds)
    precision = precision_score(truths, preds)
    recall = recall_score(truths, preds)

    return f1, precision, recall


class WeightedBCELoss():
    def __init__(self, zero_weight, one_weight):
        self.zero_weight = zero_weight
        self.one_weight = one_weight
    
    def loss(self, output, target):
        loss = self.one_weight * (target * torch.log(output)) + \
               self.zero_weight * ((1 - target) * torch.log(1 - output))
        
        # return torch.neg(torch.mean(loss))
        return torch.neg(torch.sum(loss))