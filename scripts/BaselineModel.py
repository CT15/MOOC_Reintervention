import torch
import torch.nn as nn

from model_utils import create_emb_layer

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
        # assert hidden[0].size() == torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # hidden_state
        # assert hidden[1].size() ==  torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # cell_state

        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #assert lstm_out.size() == torch.Size([seq_len, BATCH_SIZE, 1 * self.hidden_dim])
        # n_layer * 1 (1 for unidirectional and 2 for bidirectional lstm)
        # assert hidden[0].size() == torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # hidden_state
        # assert hidden[1].size() ==  torch.Size([self.n_layers * 1, BATCH_SIZE, self.hidden_dim]) # cell_state

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # put into linear layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))

        return hidden