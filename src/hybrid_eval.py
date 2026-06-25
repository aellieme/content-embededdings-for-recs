#!/usr/bin/env python3
import sys
import gc
import signal
import traceback
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from new_basic_feature import add_watch_ratio
from get_data import load_data

sys.stdout.reconfigure(line_buffering=True)

def signal_handler(sig, frame):
    print(f"\n[!!!] Получен сигнал {sig}. Завершение.")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ---------- 1. Загрузка модели ----------
try:
    print("1. Загрузка baseline_factors.pkl ...")
    with open("baseline_factors.pkl", "rb") as f:
        data = pickle.load(f)
    user_factors = data['user_factors']
    item_factors = data['item_factors']
    train_matrix = data['train_matrix']
    df_val = data['df_val']
    emb_matrix = data['emb_matrix']
    items_meta = data['items_meta']
    # Синхронизация размерности emb_matrix с item_factors
    emb_matrix = emb_matrix[:item_factors.shape[0]]
    print("   OK")
except Exception as e:
    print("ОШИБКА при загрузке:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------- 2. Загрузка df_train ----------
try:
    print("2. Загрузка тренировочных данных (df_train) ...")
    df_train, _, _, _, _, _ = load_data("up0.01_ir0.01")
    df_train = add_watch_ratio(df_train, items_meta)
    print("   OK")
except Exception as e:
    print("ОШИБКА при загрузке df_train:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------- 3. Вычисление эмбеддингов пользователей ----------
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

try:
    print("3. Вычисление user_embeddings (группировка)...")
    user_embeddings = compute_user_embeddings(df_train, emb_matrix)
    print("   OK")
except Exception as e:
    print("ОШИБКА при вычислении user_embeddings:", e)
    traceback.print_exc()
    sys.exit(1)

del df_train
gc.collect()
print("   (память от df_train освобождена)")

# ---------- 4. Нормализация ----------
def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, where=norm!=0, out=np.zeros_like(x))

try:
    print("4. Нормализация эмбеддингов...")
    user_embeds_norm = normalize_rows(user_embeddings)
    item_embeds_norm = normalize_rows(emb_matrix)
    del user_embeddings
    gc.collect()
    print("   OK")
except Exception as e:
    print("ОШИБКА при нормализации:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------- 5. Гибридный предиктор ----------
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

# ---------- 6. Функция оценки ----------
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

# ---------- 7. Baseline ----------
try:
    print("5. Оценка baseline ALS (1000 пользователей)...")
    user_ids_sample = df_val['user_idx'].unique()[:1000]
    recall_base, ndcg_base = evaluate_numpy(user_ids_sample, user_factors, item_factors,
                                            train_matrix, df_val, k=10)
    print(f"   Baseline ALS: Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")
except Exception as e:
    print("ОШИБКА при оценке baseline:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------- 8. Подбор alpha ----------
try:
    print("6. Подбор alpha (200 пользователей, 5 значений)...")
    val_sample = df_val.sample(n=200, random_state=42)
    sample_user_ids = val_sample['user_idx'].unique()
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_alpha, best_ndcg = 0, 0

    for alpha in tqdm(alphas, desc="Tuning alpha"):
        _, ndcg = evaluate_numpy(sample_user_ids, user_factors, item_factors,
                                 train_matrix, val_sample, k=10,
                                 user_embeds_norm=user_embeds_norm,
                                 item_embeds_norm=item_embeds_norm,
                                 alpha=alpha)
        print(f"   alpha={alpha:.2f} -> NDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = alpha
    print(f"   Лучший alpha = {best_alpha:.2f} (NDCG={best_ndcg:.4f})")
except Exception as e:
    print("ОШИБКА при подборе alpha:", e)
    traceback.print_exc()
    sys.exit(1)

# ---------- 9. Финальная оценка гибрида ----------
try:
    print("7. Финальная оценка гибрида (2000 пользователей)...")
    val_full = df_val.sample(n=2000, random_state=42)
    user_ids_full = val_full['user_idx'].unique()
    recall_hybrid, ndcg_hybrid = evaluate_numpy(user_ids_full, user_factors, item_factors,
                                                train_matrix, val_full, k=10,
                                                user_embeds_norm=user_embeds_norm,
                                                item_embeds_norm=item_embeds_norm,
                                                alpha=best_alpha)
    print(f"   Hybrid (alpha={best_alpha:.2f}): Recall@10={recall_hybrid:.4f}, NDCG@10={ndcg_hybrid:.4f}")
except Exception as e:
    print("ОШИБКА при финальной оценке гибрида:", e)
    traceback.print_exc()
    sys.exit(1)

print("=== Готово! ===")