import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from new_basic_feature import add_watch_ratio  # только если нужно

# Загружаем сохранённые данные
with open("baseline_factors.pkl", "rb") as f:
    data = pickle.load(f)

user_factors = data['user_factors']
item_factors = data['item_factors']
train_matrix = data['train_matrix']
df_val = data['df_val']
emb_matrix = data['emb_matrix']
items_meta = data['items_meta']

# Если df_val уже содержит watch_ratio, то можно пропустить, иначе добавить
# df_val = add_watch_ratio(df_val, items_meta)  # если не было

# ---------- Контентные эмбеддинги пользователей (также чисто numpy) ----------
# Загружаем df_train (или сохраняли его тоже)
from get_data import load_data
df_train, _, _, _, _, _ = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)

user_embeddings = np.zeros((user_factors.shape[0], emb_matrix.shape[1]))
for user in tqdm(range(user_factors.shape[0]), desc="User embeds"):
    user_items = df_train[df_train['user_idx'] == user][['item_idx', 'watch_ratio']]
    if len(user_items) > 0:
        weights = user_items['watch_ratio'].values.reshape(-1, 1)
        item_embeds = emb_matrix[user_items['item_idx'].values]
        user_embeddings[user] = np.average(item_embeds, axis=0, weights=weights.flatten())

def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, where=norm!=0, out=np.zeros_like(x))

user_embeds_norm = normalize_rows(user_embeddings)
item_embeds_norm = normalize_rows(emb_matrix)

# ---------- Функция предсказания (без implicit) ----------
def hybrid_predict_numpy(user_ids, user_factors, item_factors, train_matrix,
                         alpha, user_embeds_norm, item_embeds_norm, N=10):
    hybrid_scores = []
    for user_idx in user_ids:
        als_vec = user_factors[user_idx] @ item_factors.T
        cont_vec = user_embeds_norm[user_idx] @ item_embeds_norm.T
        hybrid = alpha * als_vec + (1 - alpha) * cont_vec
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores

# ---------- Функция оценки ----------
def evaluate_numpy(user_ids, user_factors, item_factors, train_matrix, val_df,
                   k=10, user_embeds_norm=None, item_embeds_norm=None, alpha=None):
    if alpha is None:
        # чистый ALS: для каждого пользователя топ-N по скалярному произведению
        recommendations = []
        for user in user_ids:
            scores = user_factors[user] @ item_factors.T
            seen = train_matrix[user].indices
            scores[seen] = -np.inf
            top = np.argsort(scores)[::-1][:k]
            recommendations.append(top)
    else:
        recommendations = hybrid_predict_numpy(user_ids, user_factors, item_factors,
                                               train_matrix, alpha,
                                               user_embeds_norm, item_embeds_norm, N=k)
    user_true_items = defaultdict(list)
    for _, row in val_df.iterrows():
        user_true_items[row['user_idx']].append(row['item_idx'])
    recalls, ndcgs = [], []
    for i, user in enumerate(user_ids):
        true = set(user_true_items[user])
        pred = recommendations[i]
        if not true:
            continue
        hit = sum(1 for p in pred if p in true)
        recall = hit / len(true)
        recalls.append(recall)
        dcg = sum(1 / np.log2(pos+2) for pos, p in enumerate(pred) if p in true)
        idcg = sum(1 / np.log2(pos+2) for pos in range(min(len(true), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    return np.mean(recalls), np.mean(ndcgs)

# ---------- Оценка ----------
user_ids = df_val['user_idx'].unique()

# Baseline
recall_base, ndcg_base = evaluate_numpy(user_ids, user_factors, item_factors,
                                        train_matrix, df_val, k=10)
print(f"Baseline ALS: Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

# Подбор alpha (на подвыборке)
val_sample = df_val.sample(n=5000, random_state=42)
sample_user_ids = val_sample['user_idx'].unique()
alphas = np.linspace(0, 1, 11)
best_alpha, best_ndcg = 0, 0
for alpha in tqdm(alphas, desc="Tuning alpha"):
    _, ndcg = evaluate_numpy(sample_user_ids, user_factors, item_factors,
                             train_matrix, val_sample, k=10,
                             user_embeds_norm=user_embeds_norm,
                             item_embeds_norm=item_embeds_norm,
                             alpha=alpha)
    print(f"alpha={alpha:.1f} -> NDCG@10={ndcg:.4f}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_alpha = alpha

# Гибрид на полной валидации
recall_hybrid, ndcg_hybrid = evaluate_numpy(user_ids, user_factors, item_factors,
                                            train_matrix, df_val, k=10,
                                            user_embeds_norm=user_embeds_norm,
                                            item_embeds_norm=item_embeds_norm,
                                            alpha=best_alpha)
print(f"Hybrid (alpha={best_alpha:.1f}): Recall@10={recall_hybrid:.4f}, NDCG@10={ndcg_hybrid:.4f}")