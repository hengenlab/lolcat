from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, Set2Set


class TimeEmbedding(nn.Module):
    def __init__(self, emb_size, emb_dim):
        super().__init__()

        self.linear = nn.Linear(1, emb_size, bias=True)
        self.embedding = nn.Embedding(emb_size, emb_dim)

    def forward(self, t):
        # t: (batch, 1)
        att = torch.softmax(self.linear(t), dim=1)
        ret = att.unsqueeze(-1) * self.embedding.weight.unsqueeze(0)
        ret = ret.mean(dim=1)
        return ret


class isiLSTM(nn.Module):
    r"""
    padding https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
    """

    def __init__(self, emb_size, emb_dim, lstm_hidden_size, classify=17, classifier_dropout_p=0.2, trial_dropout_p=0.2):
        super().__init__()

        self.time_emb = TimeEmbedding(emb_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden_size, batch_first=True, num_layers=4)

        classifier_head = lambda num_classes: nn.Sequential(nn.Linear(2 * lstm_hidden_size, 64), nn.ReLU(),
                                                            nn.Dropout(classifier_dropout_p),
                                                            nn.Linear(64, 32), nn.ReLU(),
                                                            nn.Dropout(classifier_dropout_p),
                                                            nn.Linear(32, num_classes))
        if isinstance(classify, int):
            classify = [classify]
        self.classify = classify
        self.classifier = nn.ModuleDict(
            {classify_i: classifier_head(train_dataset.num_classes[classify_i]) for classify_i in classify})

        self.set2set = Set2Set(lstm_hidden_size, processing_steps=4)
        self.trial_dropout_p = trial_dropout_p

    def dropout_trials(self, batch, ignore_id):
        if self.training:
            dropout_mask = torch.empty(batch.size(), dtype=torch.float32, device=batch.device).uniform_(0,
                                                                                                        1) < self.trial_dropout_p
            batch[dropout_mask] = ignore_id
            return batch
        else:
            return batch

    def forward(self, data):
        x, trial_id, batch = data.x, data.trial_id, data.batch

        # compute temporal embedding (spike wise)
        t = self.time_emb(x.view(-1, 1))

        # feed to lstm and compute sequence embedding (trial wise)
        t, seq_mask = to_dense_batch(t, batch * 100 + trial_id)

        seq_lengths = seq_mask.sum(dim=1)
        # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # t = t[perm_idx]

        packed_t = pack_padded_sequence(t, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(packed_t)

        seq_embedding = ct[-1]

        # global pooling (cell wise)
        # seq_embedding = seq_embedding.view(data.num_graphs, 100, -1)
        ignore_id = data.num_graphs
        trial_batch = torch.repeat_interleave(torch.arange(data.num_graphs, dtype=torch.long, device=x.device), 100)

        trial_batch = self.dropout_trials(trial_batch, ignore_id)
        cell_embedding = self.set2set(seq_embedding, trial_batch)[:data.num_graphs]

        # classification
        out = []
        for classify_i in self.classify:
            logits = torch.log_softmax(self.classifier[classify_i](cell_embedding), dim=1)
            out.append(logits)
        return out