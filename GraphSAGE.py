import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
import random
from typing import Dict, List, Tuple

DATA_PATH = './data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_VALUE = 20

CONFIG = {
    'embed_size': 64,
    'layers': [64, 64],
    'decay': 1e-4,
    'learning_rate': 0.001,
    'batch_size': 2048,
    'epochs': 1000,
    'eval_freq': 50,
    'patience': 200
}

class Data:
    def __init__(self, path):
        self.path = path
        self.n_users, self.n_items = self._count_entities()
        self.train_data = self._load_ratings(f'{path}/train.txt')
        self.test_data = self._load_ratings(f'{path}/test.txt')
        self.adj_mat = self._normalize_adj_matrix().to(DEVICE)

    def _count_entities(self):
        users, items = set(), set()
        for file in [f'{self.path}/train.txt', f'{self.path}/test.txt']:
            if os.path.exists(file):
                with open(file) as f:
                    for line in f:
                        parts = [int(i) for i in line.split()]
                        if not parts: continue
                        users.add(parts[0])
                        items.update(parts[1:])
        return max(users) + 1, max(items) + 1

    def _load_ratings(self, filename):
        data = {}
        if os.path.exists(filename):
            with open(filename) as f:
                for line in f:
                    parts = [int(i) for i in line.split()]
                    if len(parts) > 1:
                        data[parts[0]] = parts[1:]
        return data

    def _normalize_adj_matrix(self):
        rows, cols = [], []
        for u, items in self.train_data.items():
            for i in items:
                rows.append(u)
                cols.append(i)

        R = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_users, self.n_items))
        adj = sp.bmat([[None, R], [R.T, None]], format='csr')

        # Simetrična normalizacija: D^-0.5 * A * D^-0.5
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=rowsum != 0)
        d_inv[np.isinf(d_inv)] = 0.
        D = sp.diags(d_inv)

        A_hat = D @ adj @ D
        A_hat = A_hat.tocoo()

        indices = torch.from_numpy(np.vstack((A_hat.row, A_hat.col)).astype(np.int64))
        values = torch.from_numpy(A_hat.data.astype(np.float32))
        return torch.sparse_coo_tensor(indices, values, torch.Size(A_hat.shape))

    def sample_batch(self, batch_size):
        user_list = list(self.train_data.keys())
        users_sampled = random.choices(user_list, k=batch_size)
        pos_items, neg_items = [], []

        for u in users_sampled:
            pos = random.choice(self.train_data[u])
            while True:
                neg = random.randrange(self.n_items)
                if neg not in self.train_data[u]:
                    break
            pos_items.append(pos)
            neg_items.append(neg)

        return (torch.LongTensor(users_sampled).to(DEVICE),
                torch.LongTensor(pos_items).to(DEVICE),
                torch.LongTensor(neg_items).to(DEVICE))

class GraphSAGE(nn.Module):
    def __init__(self, n_users, n_items, adj_mat, config):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.adj_mat = adj_mat
        self.decay = config['decay']

        self.embedding = nn.Embedding(n_users + n_items, config['embed_size'])
        nn.init.xavier_uniform_(self.embedding.weight)

        self.layers = [config['embed_size']] + config['layers']
        self.linears = nn.ModuleList([
            nn.Linear(self.layers[i] * 2, self.layers[i+1])
            for i in range(len(self.layers)-1)
        ])

    def forward(self, users=None, pos_items=None, neg_items=None):
        h = self.embedding.weight
        all_h = [h]

        for layer in self.linears:
            neigh = torch.sparse.mm(self.adj_mat, h)
            h = layer(torch.cat([h, neigh], dim=1))
            h = F.leaky_relu(h)
            h = F.normalize(h, p=2, dim=1)
            all_h.append(h)

        final_embeddings = torch.mean(torch.stack(all_h, dim=0), dim=0)
        u_emb = final_embeddings[:self.n_users]
        i_emb = final_embeddings[self.n_users:]

        if users is not None:
            u = u_emb[users]
            pos = i_emb[pos_items]
            neg = i_emb[neg_items]

            pos_scores = torch.sum(u * pos, dim=1)
            neg_scores = torch.sum(u * neg, dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            l2_reg = (u.norm(2)**2 + pos.norm(2)**2 + neg.norm(2)**2) / (2 * len(users))
            return bpr_loss + self.decay * l2_reg

        return u_emb, i_emb

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model()
        recalls, ndcgs = [], []

        for u, items_true in data.test_data.items():
            if u >= data.n_users: continue

            scores = torch.matmul(u_emb[u], i_emb.T)

            train_items = data.train_data.get(u, [])
            scores[train_items] = -float('inf')

            _, topk_indices = torch.topk(scores, K_VALUE)
            recs = topk_indices.cpu().numpy()

            pos_set = set(items_true)
            rec_set = set(recs)
            hit_count = len(pos_set & rec_set)

            recalls.append(hit_count / len(pos_set))

            dcg = sum([1/np.log2(i+2) for i, r in enumerate(recs) if r in pos_set])
            idcg = sum([1/np.log2(i+2) for i in range(min(len(pos_set), K_VALUE))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(recalls), np.mean(ndcgs)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Greška: Mapa {DATA_PATH} ne postoji.")
        return

    data = Data(DATA_PATH)
    model = GraphSAGE(data.n_users, data.n_items, data.adj_mat, CONFIG).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    best_recall = 0
    patience_counter = 0

    print("Započinjem trening...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        users, pos, neg = data.sample_batch(CONFIG['batch_size'])

        optimizer.zero_grad()
        loss = model(users, pos, neg)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % CONFIG['eval_freq'] == 0:
            recall, ndcg = evaluate(model, data)
            print(f"Epoch {epoch+1:04d} | Loss: {loss.item():.4f} | Recall@{K_VALUE}: {recall:.4f} | NDCG@{K_VALUE}: {ndcg:.4f}")

            if recall > best_recall:
                best_recall = recall
                patience_counter = 0
            else:
                patience_counter += CONFIG['eval_freq']

            if patience_counter >= CONFIG['patience']:
                print("Early stopping aktiviran.")
                break

if __name__ == "__main__":
    main()
