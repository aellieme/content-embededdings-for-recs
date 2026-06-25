import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from als_functions import hybrid_predict
from new_basic_feature import add_watch_ratio  

# Загружаем сохранённые объекты
with open("baseline_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data['model']
train_matrix = data['train_matrix']
df_val = data['df_val']
emb_matrix = data['emb_matrix']

from get_data import load_data
df_train, _, _, _, _, _ = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, data['items_meta'])  # items_meta есть в сохранённом объекте

user_embeddings = np.zeros((train_matrix.shape[0], emb_matrix.shape[1]))
for user in tqdm(range(train_matrix.shape[0]), desc="Computing user embeddings"):
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

# ---------- Оценочная функция ----------
def evaluate_model(model, val_df, train_matrix, k=10,
                   user_embeds_norm=None, item_embeds_norm=None, alpha=None):
    user_ids = val_df['user_idx'].unique()
    if alpha is None:
        recommendations = model.recommend(user_ids, model.item_factors, N=k, filter_already_liked_items=True)
    else:
        recommendations = hybrid_predict(model, user_ids, train_matrix, alpha,
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

# ---------- Baseline (на уже обученной модели) ----------
recall_base, ndcg_base = evaluate_model(model, df_val, train_matrix, k=10)
print(f"Baseline ALS: Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

# ---------- Подбор alpha (на подвыборке для скорости) ----------
val_sample = df_val.sample(n=5000, random_state=42)  # можно увеличить, если позволяет время
alphas = np.linspace(0, 1, 11)
best_alpha = 0
best_ndcg = 0
for alpha in tqdm(alphas, desc="Tuning alpha"):
    _, ndcg = evaluate_model(model, val_sample, train_matrix, k=10,
                             user_embeds_norm=user_embeds_norm,
                             item_embeds_norm=item_embeds_norm,
                             alpha=alpha)
    print(f"alpha={alpha:.1f} -> NDCG@10={ndcg:.4f}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_alpha = alpha

print(f"\nBest alpha = {best_alpha:.1f} with NDCG@10 = {best_ndcg:.4f}")

recall_hybrid, ndcg_hybrid = evaluate_model(model, df_val, train_matrix, k=10,
                                            user_embeds_norm=user_embeds_norm,
                                            item_embeds_norm=item_embeds_norm,
                                            alpha=best_alpha)
print(f"Hybrid (alpha={best_alpha:.1f}): Recall@10={recall_hybrid:.4f}, NDCG@10={ndcg_hybrid:.4f}")