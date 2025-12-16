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
    'embed_size': 256,
    'layers': [256],
    'decay': 0.0001,
    'learning_rate': 0.0005,
    'batch_size': 8192,
    'epochs': 5000,
    'eval_freq': 100,
    'patience': 300
}

class Data:
    def __init__(self, path_to_data: str):
        self.path = path_to_data
        print("Loading data and preprocessing...")

        self.n_users, self.n_items = self._count_entities()
        self.train_data: Dict[int, List[int]] = self._load_ratings(f'{self.path}/train.txt')
        self.test_data: Dict[int, List[int]] = self._load_ratings(f'{self.path}/test.txt')

        if self.n_users == 0 or self.n_items == 0:
            raise ValueError("Nije moguće učitati korisnike/stavke. Provjerite train.txt.")

        self.R = self._create_sparse_matrix()
        self.adj_mat = self._normalize_adj_matrix().to(DEVICE)

        self.all_items = set(range(self.n_items))

    def _count_entities(self) -> Tuple[int, int]:
        print("Warning: Inferring counts from train.txt...")
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
                    if len(line) == 0:
                        continue
                    line = line.strip('\n')
                    items = [int(i) for i in line.split() if i.isdigit()]
                    if not items:
                        continue
                    u_id = items[0]
                    pos_i_ids = items[1:]
                    ratings[u_id] = pos_i_ids
        except FileNotFoundError:
            print(f"File not found: {filename}")
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

        row = torch.LongTensor(A_hat.row)
        col = torch.LongTensor(A_hat.col)
        index = torch.stack([row, col])
        data = torch.FloatTensor(A_hat.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(A_hat.shape))

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

class NGCF(nn.Module):
    def __init__(self, data_config: Dict, adj_mat: torch.sparse.FloatTensor, layers: List[int], decay: float):
        super(NGCF, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_nodes = self.n_users + self.n_items
        self.adj_mat = adj_mat
        self.layers = [data_config['embed_size']] + layers
        self.decay = decay

        self.embedding = nn.Embedding(self.n_nodes, self.layers[0])
        nn.init.xavier_uniform_(self.embedding.weight.data)

        self.weights = self._init_weights()

    def _init_weights(self) -> nn.ModuleDict:
        weights = nn.ModuleDict()
        for k in range(len(self.layers) - 1):
            weights[f'W1_{k}'] = nn.Linear(self.layers[k], self.layers[k + 1], bias=True)
            weights[f'W2_{k}'] = nn.Linear(self.layers[k], self.layers[k + 1], bias=True)
            nn.init.xavier_uniform_(weights[f'W1_{k}'].weight)
            nn.init.xavier_uniform_(weights[f'W2_{k}'].weight)
        return weights

    def forward(self, users: torch.Tensor, pos_items=None, neg_items=None):
        all_embeddings = self.embedding.weight
        all_h = [all_embeddings]

        reg_params = []

        for k in range(len(self.layers) - 1):
            W1 = self.weights[f'W1_{k}']
            W2 = self.weights[f'W2_{k}']
            h = all_h[k]

            ego_part = W1(h)
            agg_h = torch.sparse.mm(self.adj_mat, h)
            collab_part = torch.mul(h, agg_h)
            collab_part = W2(collab_part)

            h_new = F.leaky_relu(ego_part + collab_part, negative_slope=0.2)
            h_new = F.normalize(h_new, p=2, dim=1)

            all_h.append(h_new)

            reg_params.append(W1.weight)
            reg_params.append(W2.weight)

        final_embeddings = torch.cat(all_h, dim=1)
        u_g_embeddings = final_embeddings[:self.n_users]
        i_g_embeddings = final_embeddings[self.n_users:]

        if pos_items is not None and neg_items is not None:
            u_emb = u_g_embeddings[users]
            pos_i_emb = i_g_embeddings[pos_items]
            neg_i_emb = i_g_embeddings[neg_items]

            pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)
            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

            l2_reg_emb = (
                (1 / 2) * (
                    self.embedding.weight[users].norm(2).pow(2) +
                    self.embedding.weight[pos_items].norm(2).pow(2) +
                    self.embedding.weight[neg_items].norm(2).pow(2)
                ) / float(len(users))
            )

            l2_reg_weights = sum(p.norm(2).pow(2) for p in reg_params) / float(len(users))
            reg_loss = l2_reg_emb + l2_reg_weights

            loss = bpr_loss + self.decay * reg_loss
            return loss, pos_scores, neg_scores
        else:
            return u_g_embeddings, i_g_embeddings

def get_ndcg(rank_list: List[int], pos_items: List[int]) -> float:
    if not pos_items:
        return 0.0

    relevance = [1 if item in pos_items else 0 for item in rank_list]

    dcg = 0.0
    ideal_relevance = sorted([1] * len(pos_items), reverse=True)
    idcg = 0.0

    for i in range(len(relevance)):
        discount = np.log2(i + 2)
        dcg += relevance[i] / discount
        if i < len(ideal_relevance):
            idcg += ideal_relevance[i] / discount

    return dcg / idcg if idcg > 0.0 else 0.0

def get_recall(rank_list: List[int], pos_items: List[int]) -> float:
    if not pos_items:
        return 0.0
    hit = len(set(rank_list) & set(pos_items))
    return hit / len(pos_items)

def evaluate_model(model: NGCF, data_loader: Data, K_values: List[int]) -> Dict[int, Tuple[float, float]]:
    model.eval()

    with torch.no_grad():
        user_embeds, item_embeds = model(None, None, None)

    results = {k: {'recall': [], 'ndcg': []} for k in K_values}
    test_users = list(data_loader.test_data.keys())
    eval_batch_size = 1024

    for i in range(0, len(test_users), eval_batch_size):
        batch_users_ids = test_users[i:i + eval_batch_size]
        u_ids_tensor = torch.LongTensor(batch_users_ids).to(DEVICE)
        u_emb = user_embeds[u_ids_tensor]

        scores = torch.matmul(u_emb, item_embeds.transpose(0, 1))

        for idx, u in enumerate(batch_users_ids):
            seen_items = data_loader.train_data.get(u, [])
            mask = torch.LongTensor(seen_items).to(DEVICE)
            scores[idx].index_fill_(0, mask, float('-inf'))

        _, topk_indices = torch.topk(scores, max(K_values), dim=1)

        for idx, u in enumerate(batch_users_ids):
            pos_items = data_loader.test_data.get(u, [])
            if not pos_items:
                continue

            recommended_items = topk_indices[idx].cpu().numpy().tolist()

            for k in K_values:
                k_recommended = recommended_items[:k]
                results[k]['recall'].append(get_recall(k_recommended, pos_items))
                results[k]['ndcg'].append(get_ndcg(k_recommended, pos_items))

    final_results = {}
    for k in K_values:
        avg_recall = np.mean(results[k]['recall']) if results[k]['recall'] else 0.0
        avg_ndcg = np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0.0
        final_results[k] = (avg_recall, avg_ndcg)

    return final_results

def print_results_table(results: Dict, best_epoch: int, dataset_name: str = "Vaš_Dataset"):
    header = f"| Model | Dataset | Recall@{K_VALUES[0]} | NDCG@{K_VALUES[0]} | Najbolja Epoha |"
    separator = "|---|---|---|---|---|"

    print("\n" + "=" * 80)
    print("✨ Optimalni NGCF Rezultati (sačuvan najbolji model) ✨")
    print(header)
    print(separator)

    k = K_VALUES[0]
    recall = results[k][0]
    ndcg = results[k][1]

    row = f"| **NGCF** | {dataset_name} | **{recall:.4f}** | **{ndcg:.4f}** | **{best_epoch}** |"
    print(row)
    print("=" * 80)

def main():
    config = CONFIG

    try:
        data_loader = Data(DATA_PATH)
    except Exception as e:
        print(f"FATALNA GREŠKA PRI UČITAVANJU PODATAKA: {e}")
        return

    config['n_users'] = data_loader.n_users
    config['n_items'] = data_loader.n_items
    adj_mat = data_loader.adj_mat

    total_concatenated_embeds = config['embed_size'] * (len(config['layers']) + 1)

    print(f"\n--- Data Summary ---")
    print(f"Total Users: {config['n_users']}, Total Items: {config['n_items']}")
    print(f"Embedding Size: {config['embed_size']}, Layers: {config['layers']}, Concatenated Embeddings: {total_concatenated_embeds}")
    print(f"Learning Rate: {config['learning_rate']}, Decay: {config['decay']}")
    print(f"Maksimalne Epohe: {config['epochs']}, Early Stopping Patience: {config['patience']}")
    print(f"Training on device: {DEVICE}")
    print("--------------------")

    model = NGCF(config, adj_mat, config['layers'], config['decay']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_recall = 0.0
    best_ndcg = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state_dict = model.state_dict()

    print("\n--- Starting Training (with Early Stopping) ---")
    start_time = time.time()

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        users, pos_items, neg_items = data_loader.sample_batch(config['batch_size'])
        loss, _, _ = model(users, pos_items, neg_items)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:4d}/{config['epochs']}, Loss: {loss.item():.6f}")

        if (epoch + 1) % config['eval_freq'] == 0:
            model.eval()
            metrics = evaluate_model(model, data_loader, K_VALUES)
            recall = metrics[K_VALUES[0]][0]
            ndcg = metrics[K_VALUES[0]][1]
            print(f"--- EVAL: Recall@{K_VALUES[0]}: {recall:.4f}, NDCG@{K_VALUES[0]}: {ndcg:.4f} ---")

            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                best_epoch = epoch + 1
                patience_counter = 0
                best_state_dict = model.state_dict()
                print(">>> New Best Model Saved <<<")
            else:
                patience_counter += config['eval_freq']

            if patience_counter >= config['patience']:
                print(f"\n--- Early Stopping Triggered at Epoch {epoch + 1} ---")
                break

            model.train()

    end_time = time.time()
    print(f"\nTraining Complete in {end_time - start_time:.2f} seconds.")

    print("\n--- Prikazivanje Najboljeg Rezultata ---")

    model.load_state_dict(best_state_dict)
    model.eval()

    final_best_metrics = {K_VALUES[0]: (best_recall, best_ndcg)}
    print_results_table(final_best_metrics, best_epoch, dataset_name="Dataset")

if __name__ == '__main__':
    main()
