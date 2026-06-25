import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import gc
from new_basic_feature import add_watch_ratio
from get_data import load_data

# ---------- Загрузка сохранённых данных ----------
with open("baseline_factors.pkl", "rb") as f:
    data = pickle.load(f)

user_factors = data['user_factors']
item_factors = data['item_factors']
train_matrix = data['train_matrix']
df_val = data['df_val']
emb_matrix = data['emb_matrix']
items_meta = data['items_meta']

# ---------- Загрузка тренировочных данных ----------
df_train, _, _, _, _, _ = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)

# ---------- Векторизованное вычисление эмбеддингов пользователей ----------
def compute_user_embeddings(df_train, emb_matrix):
    grouped = df_train.groupby('user_idx')
    user_embeds = []
    for user, group in tqdm(grouped, desc="User embeds", total=len(grouped)):
        item_indices = group['item_idx'].values
        weights = group['watch_ratio'].values
        if len(item_indices) > 0:
            emb = np.average(emb_matrix[item_indices], axis=0, weights=weights)
        else:
            emb = np.zeros(emb_matrix.shape[1])
        user_embeds.append(emb)
    return np.array(user_embeds)

user_embeddings = compute_user_embeddings(df_train, emb_matrix)

# Освобождаем память от df_train (больше не нужен)
del df_train
gc.collect()

# ---------- Нормализация ----------
def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, where=norm!=0, out=np.zeros_like(x))

user_embeds_norm = normalize_rows(user_embeddings)
item_embeds_norm = normalize_rows(emb_matrix)

# Освобождаем исходный массив (уже не нужен)
del user_embeddings
gc.collect()

# ---------- Гибридный предиктор (без implicit) ----------
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

# ---------- Оценка baseline (на подвыборке для экономии времени) ----------
# Для baseline берём только 10000 пользователей (можно и всех, но для скорости)
user_ids_sample = df_val['user_idx'].unique()[:10000]  # или использовать все, если позволяет время
recall_base, ndcg_base = evaluate_numpy(user_ids_sample, user_factors, item_factors,
                                        train_matrix, df_val, k=10)
print(f"Baseline ALS (sample 10k): Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

# ---------- Подбор alpha на очень маленькой подвыборке ----------
val_sample = df_val.sample(n=500, random_state=42)   # критически мало для памяти
sample_user_ids = val_sample['user_idx'].unique()
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]                 # всего 5 значений
best_alpha, best_ndcg = 0, 0

for alpha in tqdm(alphas, desc="Tuning alpha"):
    _, ndcg = evaluate_numpy(sample_user_ids, user_factors, item_factors,
                             train_matrix, val_sample, k=10,
                             user_embeds_norm=user_embeds_norm,
                             item_embeds_norm=item_embeds_norm,
                             alpha=alpha)
    print(f"alpha={alpha:.2f} -> NDCG@10={ndcg:.4f}")
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        best_alpha = alpha

# ---------- Гибрид на полной валидации (или на большей подвыборке) ----------
# Для финальной оценки можно взять всех, но если память не позволяет – ограничьтесь 20k
val_full = df_val  # или df_val.sample(n=20000, random_state=42)
user_ids_full = val_full['user_idx'].unique()
recall_hybrid, ndcg_hybrid = evaluate_numpy(user_ids_full, user_factors, item_factors,
                                            train_matrix, val_full, k=10,
                                            user_embeds_norm=user_embeds_norm,
                                            item_embeds_norm=item_embeds_norm,
                                            alpha=best_alpha)
print(f"Hybrid (alpha={best_alpha:.2f}): Recall@10={recall_hybrid:.4f}, NDCG@10={ndcg_hybrid:.4f}")