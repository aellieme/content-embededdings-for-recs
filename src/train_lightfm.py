import pickle
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.evaluation import recall_at_k, ndcg_at_k
from lightfm.cross_validation import random_train_test_split

from get_data import load_data
from new_basic_feature import add_watch_ratio

# Загрузка данных (используем ту же подвыборку)
df_train, df_val, items_meta, emb_matrix, user_map, item_map = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)
df_val = add_watch_ratio(df_val, items_meta)

# Построим матрицу user-item (разреженную)
def build_interaction_matrix(df, user_col='user_idx', item_col='item_idx', rating_col='watch_ratio'):
    rows = df[user_col].values
    cols = df[item_col].values
    data = df[rating_col].values.astype(np.float32)
    return csr_matrix((data, (rows, cols)), shape=(df[user_col].max()+1, df[item_col].max()+1))

train_matrix = build_interaction_matrix(df_train)
val_matrix = build_interaction_matrix(df_val)  # для оценки

# Контентные признаки
# Пользовательские признаки: age, gender, geo (one-hot или embedding)
# Для простоты сделаем one-hot через pandas get_dummies
user_features = df_train[['user_idx', 'age', 'gender', 'geo']].drop_duplicates('user_idx').set_index('user_idx')
user_features = pd.get_dummies(user_features, columns=['age', 'gender', 'geo'])
# Приводим к разреженному виду
user_features_csr = csr_matrix(user_features.values.astype(np.float32))

# Предметные признаки: author_id, duration
item_features = items_meta[['item_id', 'author_id', 'duration']].copy()
item_features['item_idx'] = item_features['item_id'].map(item_map)
item_features = item_features.dropna().set_index('item_idx')
item_features = pd.get_dummies(item_features, columns=['author_id'])
# duration можно нормализовать
item_features['duration'] = (item_features['duration'] - item_features['duration'].mean()) / item_features['duration'].std()
item_features_csr = csr_matrix(item_features.values.astype(np.float32))

# Обучение LightFM
model = LightFM(loss='warp', learning_rate=0.05, no_components=64, random_state=42)
model.fit(train_matrix, user_features=user_features_csr, item_features=item_features_csr, epochs=30)

# Оценка
train_recall = recall_at_k(model, train_matrix, user_features=user_features_csr, item_features=item_features_csr, k=10)
val_recall = recall_at_k(model, val_matrix, user_features=user_features_csr, item_features=item_features_csr, k=10)
train_ndcg = ndcg_at_k(model, train_matrix, user_features=user_features_csr, item_features=item_features_csr, k=10)
val_ndcg = ndcg_at_k(model, val_matrix, user_features=user_features_csr, item_features=item_features_csr, k=10)

print(f"LightFM (WARP) – Train Recall@10: {train_recall:.4f}, Val Recall@10: {val_recall:.4f}")
print(f"LightFM (WARP) – Train NDCG@10: {train_ndcg:.4f}, Val NDCG@10: {val_ndcg:.4f}")