import pandas as pd
import numpy as np
from collections import defaultdict

from get_data import load_data
from new_basic_feature import add_watch_ratio
from als_functions import create_csr, train_als, compute_content_scores, hybrid_predict

# Загрузка
df_train, df_val, items_meta, emb_matrix, user_map, item_map = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)
df_val = add_watch_ratio(df_val, items_meta)

train_matrix = create_csr(df_train)

# ---------- Baseline ALS ----------
print("Training baseline ALS...")
model_baseline = train_als(train_matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True)

# ---------- Контентные эмбеддинги для пользователей ----------
user_embeddings = np.zeros((train_matrix.shape[0], emb_matrix.shape[1]))
for user in range(train_matrix.shape[0]):
    user_items = df_train[df_train['user_idx'] == user][['item_idx', 'watch_ratio']]
    if len(user_items) > 0:
        weights = user_items['watch_ratio'].values.reshape(-1, 1)
        item_embeds = emb_matrix[user_items['item_idx'].values]
        user_embeddings[user] = np.average(item_embeds, axis=0, weights=weights.flatten())

print("Computing content scores...")
content_scores = compute_content_scores(user_embeddings, emb_matrix)  # (n_users, n_items)

# ---------- Единая функция оценки ----------
def evaluate_model(model, val_df, train_matrix, k=10, content_scores=None, alpha=None):
    user_ids = val_df['user_idx'].unique()
    if alpha is None:
        recommendations = model.recommend(user_ids, model.item_factors, N=k, filter_already_liked_items=True)
    else:
        recommendations = hybrid_predict(model, user_ids, train_matrix, alpha, content_scores, N=k)
    
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

# ---------- Подбор alpha на валидации ----------
alphas = np.linspace(0, 1, 11)
best_alpha = 0
best_ndcg = 0
for alpha in alphas:
    _, ndcg = evaluate_model(model_baseline, df_val, train_matrix, k=10, content_scores=content_scores, alpha=alpha)
    print(f"alpha={alpha:.1f} -> NDCG@10={ndcg:.4f}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_alpha = alpha

print(f"\nBest alpha = {best_alpha:.1f} with NDCG@10 = {best_ndcg:.4f}")

# ---------- Оценка baseline и гибрида с лучшим alpha ----------
recall_base, ndcg_base = evaluate_model(model_baseline, df_val, train_matrix, k=10)
print(f"\nBaseline ALS: Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

recall_hybrid, ndcg_hybrid = evaluate_model(model_baseline, df_val, train_matrix, k=10,
                                            content_scores=content_scores, alpha=best_alpha)
print(f"Hybrid (alpha={best_alpha:.1f}): Recall@10={recall_hybrid:.4f}, NDCG@10={ndcg_hybrid:.4f}")