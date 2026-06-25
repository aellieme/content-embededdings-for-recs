#!/usr/bin/env python3
import sys, gc, signal, traceback, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from new_basic_feature import add_watch_ratio
from get_data import load_data

sys.stdout.reconfigure(line_buffering=True)

def signal_handler(sig, frame):
    print(f"\n[!!!] Сигнал {sig}. Завершение.")
    sys.exit(1)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- Загрузка ----------
print("1. Загрузка baseline_factors.pkl ...")
with open("baseline_factors.pkl", "rb") as f:
    data = pickle.load(f)
user_factors = data['user_factors']
item_factors = data['item_factors']
train_matrix = data['train_matrix']
df_val = data['df_val']
emb_matrix = data['emb_matrix'][:item_factors.shape[0]]  # синхронизация
items_meta = data['items_meta']
print("   OK")

print("2. Загрузка df_train ...")
df_train, _, _, _, _, _ = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)
print("   OK")

# ---------- Эмбеддинги пользователей ----------
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

print("3. Вычисление user_embeddings ...")
user_embeddings = compute_user_embeddings(df_train, emb_matrix)
del df_train
gc.collect()
print("   OK")

def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, where=norm!=0, out=np.zeros_like(x))

print("4. Нормализация ...")
user_embeds_norm = normalize_rows(user_embeddings)
item_embeds_norm = normalize_rows(emb_matrix)
del user_embeddings, emb_matrix
gc.collect()
print("   OK")

# ---------- Гибридные предикторы ----------
def hybrid_predict_sum(user_ids, user_factors, item_factors, train_matrix,
                       alpha, user_embeds_norm, item_embeds_norm, N=10):
    """alpha * ALS + (1-alpha) * content, с нормализацией скоров."""
    hybrid_scores = []
    for user_idx in user_ids:
        als_vec = user_factors[user_idx] @ item_factors.T
        cont_vec = user_embeds_norm[user_idx] @ item_embeds_norm.T
        # Нормализация Z-score (по всем предметам)
        als_vec = (als_vec - als_vec.mean()) / (als_vec.std() + 1e-8)
        cont_vec = (cont_vec - cont_vec.mean()) / (cont_vec.std() + 1e-8)
        hybrid = alpha * als_vec + (1 - alpha) * cont_vec
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores

def hybrid_predict_rank(user_ids, user_factors, item_factors, train_matrix,
                        alpha, user_embeds_norm, item_embeds_norm, N=10):
    """alpha * rank(ALS) + (1-alpha) * rank(content)."""
    hybrid_scores = []
    for user_idx in user_ids:
        als_vec = user_factors[user_idx] @ item_factors.T
        cont_vec = user_embeds_norm[user_idx] @ item_embeds_norm.T
        # Ранги (чем выше скор, тем лучше; ранг 1 = лучший)
        als_rank = np.argsort(np.argsort(-als_vec)) + 1   # 1..n_items
        cont_rank = np.argsort(np.argsort(-cont_vec)) + 1
        hybrid = alpha * als_rank + (1 - alpha) * cont_rank
        # В данном случае меньший ранг лучше, поэтому берём топ N с минимальным значением
        # Для унификации возьмём отрицание, чтобы чем меньше ранг, тем выше "скор"
        hybrid = -hybrid
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores

def hybrid_predict_multiply(user_ids, user_factors, item_factors, train_matrix,
                            alpha, user_embeds_norm, item_embeds_norm, N=10):
    """(ALS^alpha) * (content^(1-alpha)) (сглаженное умножение)."""
    hybrid_scores = []
    for user_idx in user_ids:
        als_vec = user_factors[user_idx] @ item_factors.T
        cont_vec = user_embeds_norm[user_idx] @ item_embeds_norm.T
        # Сдвиг, чтобы избежать отрицательных значений (если есть)
        als_vec = als_vec - als_vec.min() + 1e-6
        cont_vec = cont_vec - cont_vec.min() + 1e-6
        hybrid = (als_vec ** alpha) * (cont_vec ** (1 - alpha))
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores

# ---------- Оценщик ----------
def evaluate_numpy(user_ids, user_factors, item_factors, train_matrix, val_df,
                   k=10, user_embeds_norm=None, item_embeds_norm=None,
                   alpha=None, mode='sum'):
    if alpha is None:
        # чистый ALS
        recommendations = []
        for user in user_ids:
            scores = user_factors[user] @ item_factors.T
            seen = train_matrix[user].indices
            scores[seen] = -np.inf
            top = np.argsort(scores)[::-1][:k]
            recommendations.append(top)
    else:
        if mode == 'sum':
            recommendations = hybrid_predict_sum(user_ids, user_factors, item_factors,
                                                 train_matrix, alpha,
                                                 user_embeds_norm, item_embeds_norm, N=k)
        elif mode == 'rank':
            recommendations = hybrid_predict_rank(user_ids, user_factors, item_factors,
                                                  train_matrix, alpha,
                                                  user_embeds_norm, item_embeds_norm, N=k)
        elif mode == 'multiply':
            recommendations = hybrid_predict_multiply(user_ids, user_factors, item_factors,
                                                      train_matrix, alpha,
                                                      user_embeds_norm, item_embeds_norm, N=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

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

# ---------- Основные эксперименты ----------
SAMPLE_SIZE = 10000  # увеличили
val_sample = df_val.sample(n=SAMPLE_SIZE, random_state=42)
user_ids_sample = val_sample['user_idx'].unique()

# Baseline (чистый ALS)
recall_base, ndcg_base = evaluate_numpy(user_ids_sample, user_factors, item_factors,
                                        train_matrix, val_sample, k=10)
print(f"\nBaseline ALS (n={SAMPLE_SIZE}): Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

# Расширенная сетка alpha
alphas = np.linspace(0, 1, 21)
modes = ['sum', 'rank', 'multiply']

best_results = {}
for mode in modes:
    print(f"\n=== Режим: {mode} ===")
    best_alpha, best_ndcg = 0, 0
    for alpha in tqdm(alphas, desc=f"Tuning {mode}"):
        _, ndcg = evaluate_numpy(user_ids_sample, user_factors, item_factors,
                                 train_matrix, val_sample, k=10,
                                 user_embeds_norm=user_embeds_norm,
                                 item_embeds_norm=item_embeds_norm,
                                 alpha=alpha, mode=mode)
        print(f"   alpha={alpha:.2f} -> NDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = alpha
    print(f"   Лучший alpha = {best_alpha:.2f} (NDCG={best_ndcg:.4f})")
    best_results[mode] = (best_alpha, best_ndcg)

# Финальная оценка на большей выборке (можно увеличить до всех, если память позволяет)
val_full = df_val.sample(n=20000, random_state=42)  # или df_val для всех
user_ids_full = val_full['user_idx'].unique()

print("\n=== Финальная оценка на 20k пользователей ===")
for mode, (best_alpha, _) in best_results.items():
    recall, ndcg = evaluate_numpy(user_ids_full, user_factors, item_factors,
                                  train_matrix, val_full, k=10,
                                  user_embeds_norm=user_embeds_norm,
                                  item_embeds_norm=item_embeds_norm,
                                  alpha=best_alpha, mode=mode)
    print(f"{mode:10} alpha={best_alpha:.2f} -> Recall@10={recall:.4f}, NDCG@10={ndcg:.4f}")

print("\n=== Готово ===")