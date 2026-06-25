import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from implicit.evaluation import ranking_metrics_at_k
from implicit.evaluation import train_test_split as implicit_split

from get_data import load_data
from new_basic_feature import add_watch_ratio
from als_functions import create_csr, train_als, set_item_factors_from_embeddings

# Загружаем данные (используем подвыборку up0.01_ir0.01)
df_train, df_val, items_meta, emb_matrix, user_map, item_map = load_data("up0.01_ir0.01")

# Добавляем признак watch_ratio
df_train = add_watch_ratio(df_train, items_meta)
df_val = add_watch_ratio(df_val, items_meta)

# Создаём CSR-матрицу для обучения (используем watch_ratio как рейтинг)
train_matrix = create_csr(df_train)

# ---------- Базовый ALS ----------
print("Training baseline ALS...")
model_baseline = train_als(train_matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True)

# ---------- ALS с контентными эмбеддингами (инициализация) ----------
print("Training ALS with content embeddings (initialization)...")
model_content = train_als(train_matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True)
# Заменяем факторы предметов на эмбеддинги
model_content = set_item_factors_from_embeddings(model_content, emb_matrix, train_matrix.shape[1])

# ---------- Оценка на валидации ----------
def evaluate_model(model, val_df, k=10):
    """Вычисляет recall@k и NDCG@k на валидационных данных."""
    test_users = val_df['user_idx'].values
    test_items = val_df['item_idx'].values
    from collections import defaultdict
    user_true_items = defaultdict(list)
    for _, row in val_df.iterrows():
        user_true_items[row['user_idx']].append(row['item_idx'])
    user_ids = list(user_true_items.keys())
    recommendations = model.recommend(user_ids, train_matrix, N=k, filter_already_liked_items=False)
    recalls = []
    ndcgs = []
    for i, user in enumerate(user_ids):
        true = set(user_true_items[user])
        pred = recommendations[i]
        if not true:
            continue
        hit = sum(1 for p in pred if p in true)
        recall = hit / len(true)
        recalls.append(recall)
        # NDCG 
        dcg = sum(1 / np.log2(pos+2) for pos, p in enumerate(pred) if p in true)
        idcg = sum(1 / np.log2(pos+2) for pos in range(min(len(true), k)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    return np.mean(recalls), np.mean(ndcgs)

print("Evaluating baseline...")
recall_base, ndcg_base = evaluate_model(model_baseline, df_val, k=10)
print(f"Baseline: Recall@10={recall_base:.4f}, NDCG@10={ndcg_base:.4f}")

print("Evaluating content-initialized model...")
recall_cont, ndcg_cont = evaluate_model(model_content, df_val, k=10)
print(f"Content-init: Recall@10={recall_cont:.4f}, NDCG@10={ndcg_cont:.4f}")
