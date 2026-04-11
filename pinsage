import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import random
import time
from typing import Dict, List, Tuple

DATA_PATH = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_VALUES = [20]

CONFIG = {
    'embed_size': 128,
    'layers': [128, 128],
    'decay': 1e-4,
    'learning_rate': 0.001,
    'batch_size': 8192,
    'epochs': 5000,
    'eval_freq': 100,
    'patience': 300
}

class Data:
    def __init__(self, path: str):
        self.path = path

        self.n_users, self.n_items = self._count_entities()
        self.train_data = self._load_ratings(f'{path}/train.txt')
        self.test_data = self._load_ratings(f'{path}/test.txt')

        if self.n_users == 0 or self.n_items == 0:
            raise ValueError("Neuspješno učitavanje podataka.")

        self.R = self._create_sparse_matrix()
        self.adj_mat = self._normalize_adj_matrix().to(DEVICE)

        self.all_items = set(range(self.n_items))

    def _count_entities(self):
        users, items = set(), set()

        for file in [f'{self.path}/train.txt', f'{self.path}/test.txt']:
            if not os.path.exists(file):
                continue

            with open(file) as f:
                for line in f:
                    parts = [int(i) for i in line.split() if i.isdigit()]
                    if not parts:
                        continue
                    users.add(parts[0])
                    items.update(parts[1:])

        return max(users) + 1, max(items) + 1

    def _load_ratings(self, filename):
        data = {}
        if not os.path.exists(filename):
            return data

        with open(filename) as f:
            for line in f:
                items = [int(i) for i in line.split() if i.isdigit()]
                if not items:
                    continue
                data[items[0]] = items[1:]

        return data

    def _create_sparse_matrix(self):
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for u in self.train_data:
            for i in self.train_data[u]:
                R[u, i] = 1.

        return R

    def _normalize_adj_matrix(self):

        R = self.R
        RT = R.transpose()

        A = sp.bmat(
            [[sp.csr_matrix((self.n_users, self.n_users)), R],
             [RT, sp.csr_matrix((self.n_items, self.n_items))]]
        ).tocsr()

        rowsum = np.array(A.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.

        D = sp.diags(d_inv)

        A_hat = D @ A @ D
        A_hat = A_hat.tocoo()

        index = torch.LongTensor([A_hat.row, A_hat.col])
        data = torch.FloatTensor(A_hat.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(A_hat.shape))

    def sample_batch(self, batch_size):

        users, pos_items, neg_items = [], [], []
        user_list = list(self.train_data.keys())

        for _ in range(batch_size):

            u = random.choice(user_list)
            pos = random.choice(self.train_data[u])

            while True:
                neg = random.randrange(self.n_items)
                if neg not in self.train_data[u]:
                    break

            users.append(u)
            pos_items.append(pos)
            neg_items.append(neg)

        return (
            torch.LongTensor(users).to(DEVICE),
            torch.LongTensor(pos_items).to(DEVICE),
            torch.LongTensor(neg_items).to(DEVICE)
        )

class PinSage(nn.Module):

    def __init__(self, config, adj_mat, layers, decay):
        super().__init__()

        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.n_nodes = self.n_users + self.n_items

        self.adj_mat = adj_mat
        self.layers = [config['embed_size']] + layers
        self.decay = decay

        self.embedding = nn.Embedding(self.n_nodes, self.layers[0])
        nn.init.xavier_uniform_(self.embedding.weight)

        self.linears = nn.ModuleList()

        for k in range(len(self.layers) - 1):
            self.linears.append(
                nn.Linear(self.layers[k] * 2, self.layers[k + 1])
            )

    def forward(self, users, pos_items=None, neg_items=None):

        h = self.embedding.weight
        all_h = [h]

        for layer in self.linears:

            neigh = torch.sparse.mm(self.adj_mat, h)

            neigh = F.normalize(neigh)

            h_concat = torch.cat([h, neigh], dim=1)

            h = layer(h_concat)
            h = F.relu(h)
            h = F.normalize(h)

            all_h.append(h)

        final = torch.cat(all_h, dim=1)

        u_emb = final[:self.n_users]
        i_emb = final[self.n_users:]

        if pos_items is not None:

            u = u_emb[users]
            pos = i_emb[pos_items]
            neg = i_emb[neg_items]

            pos_scores = torch.sum(u * pos, dim=1)
            neg_scores = torch.sum(u * neg, dim=1)

            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            l2_reg = (
                self.embedding.weight[users].norm(2).pow(2) +
                self.embedding.weight[pos_items].norm(2).pow(2) +
                self.embedding.weight[neg_items].norm(2).pow(2)
            ) / (2 * len(users))

            loss = bpr_loss + self.decay * l2_reg

            return loss, pos_scores, neg_scores

        return u_emb, i_emb

def get_recall(rank_list, pos_items):
    if not pos_items:
        return 0.0
    return len(set(rank_list) & set(pos_items)) / len(pos_items)

def get_ndcg(rank_list, pos_items):

    if not pos_items:
        return 0.0

    relevance = [1 if i in pos_items else 0 for i in rank_list]

    dcg, idcg = 0.0, 0.0

    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)
        if i < len(pos_items):
            idcg += 1 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_model(model, data_loader):

    model.eval()

    with torch.no_grad():
        user_embeds, item_embeds = model(None, None, None)

    recalls, ndcgs = [], []

    for u in data_loader.test_data:

        scores = torch.matmul(user_embeds[u], item_embeds.T)

        seen = data_loader.train_data.get(u, [])
        scores[seen] = -1e9

        _, topk = torch.topk(scores, K_VALUES[0])
        recs = topk.cpu().tolist()

        pos_items = data_loader.test_data[u]

        recalls.append(get_recall(recs, pos_items))
        ndcgs.append(get_ndcg(recs, pos_items))

    return np.mean(recalls), np.mean(ndcgs)

def main():

    data_loader = Data(DATA_PATH)

    CONFIG['n_users'] = data_loader.n_users
    CONFIG['n_items'] = data_loader.n_items

    model = PinSage(CONFIG, data_loader.adj_mat, CONFIG['layers'], CONFIG['decay']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
    patience_counter = 0
    best_state = model.state_dict()

    print(" Training PinSage...")

    start = time.time()

    for epoch in range(CONFIG['epochs']):

        model.train()

        users, pos_items, neg_items = data_loader.sample_batch(CONFIG['batch_size'])

        optimizer.zero_grad()

        loss, _, _ = model(users, pos_items, neg_items)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % CONFIG['eval_freq'] == 0:

            recall, ndcg = evaluate_model(model, data_loader)

            print(f"Epoch {epoch+1} | Recall@20 {recall:.4f} | NDCG@20 {ndcg:.4f}")

            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                best_epoch = epoch + 1
                best_state = model.state_dict()
                patience_counter = 0
                print(" New best model")
            else:
                patience_counter += CONFIG['eval_freq']

            if patience_counter >= CONFIG['patience']:
                print(" Early stopping")
                break

    end = time.time()

    model.load_state_dict(best_state)

    print("\n FINAL RESULTS")
    print(f"Best Epoch: {best_epoch}")
    print(f"Recall@20: {best_recall:.4f}")
    print(f"NDCG@20: {best_ndcg:.4f}")
    print(f"Training time: {end-start:.2f}s")


if __name__ == '__main__':
    main()
