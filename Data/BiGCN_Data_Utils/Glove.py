import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import numpy as np
import sys
import os
import pickle

class GloveDataset:

    def __init__(self, text, n_words=200000, window_size=5, predefined_dic="../saved/emb_layer/word_dic.pkl"):
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        word_counter = Counter()
        if os.path.exists(predefined_dic):
            with open(predefined_dic, "rb") as fr:
                tokens = list(pickle.load(fr).keys())
            word_counter.update(tokens)
        word_counter.update(self._tokens)
        self._word2id = {w: i for i, (w, _) in enumerate(word_counter.most_common())}
        self._id2word = {i: w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)

        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()

        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j - i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        # Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).cuda()
        self._j_idx = torch.LongTensor(self._j_idx).cuda()
        self._xij = torch.FloatTensor(self._xij).cuda()

    def get_batches(self, batch_size):
        # Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p + batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

    def weight_func(self, x, x_max, alpha):
        wx = (x/x_max)**alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.cuda()

    def wmse_loss(self, weights, inputs, targets):
        loss = weights * F.mse_loss(inputs, targets, reduction='none')
        return torch.mean(loss).cuda()

    def train(self, dataset, N_EPOCHS = 100, saved_model="./tmp/glove_best.pkl"):

        BATCH_SIZE = 2048
        X_MAX = 100
        ALPHA = 0.75

        optimizer = torch.optim.Adam([
            {'params': self.wi.parameters(), 'lr': 2e-5},
            {'params': self.wj.parameters(), 'lr': 2e-5},
            {'params': self.bi.parameters(), 'lr': 2e-5},
            {'params': self.bj.parameters(), 'lr': 2e-5}
        ]
        )
        n_batches = int(len(dataset._xij) / BATCH_SIZE)
        loss_values = list()
        for e in range(1, N_EPOCHS + 1):
            batch_i = 0

            for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):

                batch_i += 1

                optimizer.zero_grad()

                outputs = self.forward(i_idx, j_idx)
                weights_x = self.weight_func(x_ij, X_MAX, ALPHA)
                loss = self.wmse_loss(weights_x, outputs, torch.log(x_ij))

                loss.backward()

                optimizer.step()

                loss_values.append(loss.item())

                if batch_i % 100 == 0:
                    print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches,
                                                                            np.mean(loss_values[-20:])))

            print("Saving model...")
            torch.save(
                {
                    "wi": self.wi.state_dict(),
                    "wj": self.wj.state_dict(),
                    "bi": self.bi.state_dict(),
                    "bj": self.bj.state_dict(),
                },
                saved_model
            )

    def load_data(self, pretrained_model):
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            self.wi.load_state_dict(checkpoint['embedding'])
            self.wj.load_state_dict(checkpoint['embedding'])
        else:
            print("Error, file not exsits")
            sys.exit(0)

