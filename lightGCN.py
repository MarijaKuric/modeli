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
    'embed_size': 64,
    'layers': [64, 64, 64],
    'decay': 0.0001,
    'learning_rate': 0.001,
    'batch_size': 8192,
    'epochs': 5000,
    'eval_freq': 100,
    'patience': 300
}

class Data:
    def __init__(self, path_to_data: str):
        self.path = path_to_data

        self.n_users, self.n_items = self._count_entities()
        self.train_data: Dict[int, List[int]] = self._load_ratings(f'{self.path}/train.txt')
        self.test_data: Dict[int, List[int]] = self._load_ratings(f'{self.path}/test.txt')

        if self.n_users == 0 or self.n_items == 0:
            raise ValueError("Nije moguće učitati korisnike/stavke.")

        self.R = self._create_sparse_matrix()
        self.adj_mat = self._normalize_adj_matrix().to(DEVICE)
        self.all_items = set(range(self.n_items))

    def _count_entities(self) -> Tuple[int, int]:
        temp_users, temp_items = set(), set()
        for filename in [f'{self.path}/train.txt', f'{self.path}/test.txt']:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    for line in f.readlines():
                        if not line.strip():
                            continue
                        parts = [int(i) for i in line.strip().split() if i.isdigit()]
                        if not parts:
                            continue
                        u_id = parts[0]
                        temp_users.add(u_id)
                        for i_id in parts[1:]:
                            temp_items.add(i_id)

        n_users = max(temp_users) + 1 if temp_users else 0
        n_items = max(temp_items) + 1 if temp_items else 0
        return n_users, n_items

    def _load_ratings(self, filename: str) -> Dict[int, List[int]]:
        ratings = {}
        try:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    items = [int(i) for i in line.split() if i.isdigit()]
                    if not items:
                        continue
                    ratings[items[0]] = items[1:]
        except FileNotFoundError:
            pass
        return ratings

    def _create_sparse_matrix(self) -> sp.dok_matrix:
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in self.train_data:
            for i in self.train_data[u]:
                R[u, i] = 1.
        return R

    def _normalize_adj_matrix(self) -> torch.sparse.FloatTensor:
        R = self.R
        R_T = self.R.transpose()
        A = sp.bmat(
            [[sp.csr_matrix((self.n_users, self.n_users)), R],
             [R_T, sp.csr_matrix((self.n_items, self.n_items))]],
            format='dok'
        ).tocsr()

        sum_row = np.array(A.sum(axis=1)).flatten()
        D_inv_sqrt = np.power(sum_row, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(D_inv_sqrt)

        A_hat = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()

        index = torch.stack([
            torch.LongTensor(A_hat.row),
            torch.LongTensor(A_hat.col)
        ])
        data = torch.FloatTensor(A_hat.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(A_hat.shape))

    def sample_batch(self, batch_size: int):
        users, pos_items, neg_items = [], [], []
        u_list = list(self.train_data.keys())

        for _ in range(batch_size):
            u = random.choice(u_list)
            pos_i = random.choice(self.train_data[u])

            while True:
                neg_i = random.randrange(self.n_items)
                if neg_i not in self.train_data[u]:
                    break

            users.append(u)
            pos_items.append(pos_i)
            neg_items.append(neg_i)

        return (
            torch.LongTensor(users).to(DEVICE),
            torch.LongTensor(pos_items).to(DEVICE),
            torch.LongTensor(neg_items).to(DEVICE)
        )

class LightGCN(nn.Module):
    def __init__(self, data_config: Dict, adj_mat, layers: List[int], decay: float):
        super().__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_nodes = self.n_users + self.n_items
        self.adj_mat = adj_mat
        self.n_layers = len(layers)
        self.decay = decay

        self.embedding = nn.Embedding(self.n_nodes, data_config['embed_size'])
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, users, pos_items=None, neg_items=None):
        all_embeddings = self.embedding.weight
        all_h = [all_embeddings]

        for _ in range(self.n_layers):
            all_h.append(torch.sparse.mm(self.adj_mat, all_h[-1]))

        final_embeddings = torch.stack(all_h, dim=1).mean(dim=1)

        u_g_embeddings = final_embeddings[:self.n_users]
        i_g_embeddings = final_embeddings[self.n_users:]

        if pos_items is not None and neg_items is not None:
            u_emb = u_g_embeddings[users]
            pos_i_emb = i_g_embeddings[pos_items]
            neg_i_emb = i_g_embeddings[neg_items]

            pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            l2_reg = (1 / 2) * (
                self.embedding.weight[users].norm(2).pow(2) +
                self.embedding.weight[pos_items].norm(2).pow(2) +
                self.embedding.weight[neg_items].norm(2).pow(2)
            ) / float(len(users))

            return bpr_loss + self.decay * l2_reg, pos_scores, neg_scores

        return u_g_embeddings, i_g_embeddings

def get_ndcg(rank_list, pos_items):
    if not pos_items:
        return 0.0
    relevance = [1 if i in pos_items else 0 for i in rank_list]
    dcg, idcg = 0.0, 0.0
    for i, rel in enumerate(relevance):
        discount = np.log2(i + 2)
        dcg += rel / discount
        if i < len(pos_items):
            idcg += 1 / discount
    return dcg / idcg if idcg > 0 else 0.0

def get_recall(rank_list, pos_items):
    if not pos_items:
        return 0.0
    return len(set(rank_list) & set(pos_items)) / len(pos_items)

def evaluate_model(model, data_loader, K_values):
    model.eval()
    with torch.no_grad():
        user_embeds, item_embeds = model(None, None, None)

    results = {k: {'recall': [], 'ndcg': []} for k in K_values}
    test_users = list(data_loader.test_data.keys())

    for u in test_users:
        u_emb = user_embeds[u]
        scores = torch.matmul(u_emb, item_embeds.T)
        seen = data_loader.train_data.get(u, [])
        scores[seen] = -1e9

        _, topk = torch.topk(scores, max(K_values))
        recs = topk.cpu().tolist()
        pos_items = data_loader.test_data[u]

        for k in K_values:
            results[k]['recall'].append(get_recall(recs[:k], pos_items))
            results[k]['ndcg'].append(get_ndcg(recs[:k], pos_items))

    return {
        k: (np.mean(results[k]['recall']), np.mean(results[k]['ndcg']))
        for k in K_values
    }

def main():
    data_loader = Data(DATA_PATH)
    CONFIG['n_users'] = data_loader.n_users
    CONFIG['n_items'] = data_loader.n_items

    model = LightGCN(CONFIG, data_loader.adj_mat, CONFIG['layers'], CONFIG['decay']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    best_recall, best_ndcg, best_epoch = 0.0, 0.0, 0
    patience_counter = 0
    best_state = model.state_dict()

    for epoch in range(CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()

        users, pos_items, neg_items = data_loader.sample_batch(CONFIG['batch_size'])
        loss, _, _ = model(users, pos_items, neg_items)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % CONFIG['eval_freq'] == 0:
            metrics = evaluate_model(model, data_loader, K_VALUES)
            recall, ndcg = metrics[K_VALUES[0]]

            if recall > best_recall:
                best_recall, best_ndcg = recall, ndcg
                best_epoch = epoch + 1
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += CONFIG['eval_freq']

            if patience_counter >= CONFIG['patience']:
                break

    model.load_state_dict(best_state)
    print(f"Best Epoch: {best_epoch}, Recall@{K_VALUES[0]}: {best_recall:.4f}, NDCG@{K_VALUES[0]}: {best_ndcg:.4f}")

if __name__ == '__main__':
    main()
