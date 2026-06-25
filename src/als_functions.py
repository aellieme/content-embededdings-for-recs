import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA

def create_csr(df, user_col='user_idx', item_col='item_idx', rating_col='watch_ratio'):
    rows = df[user_col].values
    cols = df[item_col].values
    data = df[rating_col].values.astype(np.float32)
    n_users = df[user_col].max() + 1
    n_items = df[item_col].max() + 1
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

def train_als(matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=use_gpu and HAS_CUDA,
        random_state=42
    )
    model.fit(matrix)
    return model

def hybrid_predict(model, user_ids, train_matrix, alpha,
                   user_embeds_norm, item_embeds_norm, N=10):
    """
    Гибрид: alpha * ALS + (1-alpha) * content_score.
    content_score = косинусное сходство (скалярное произведение нормализованных векторов).
    Фильтрует уже просмотренные предметы.
    """
    hybrid_scores = []
    for user_idx in user_ids:
        # ALS-оценка для всех предметов
        als_vec = model.user_factors[user_idx] @ model.item_factors.T
        # Контентная оценка (косинусное сходство)
        cont_vec = user_embeds_norm[user_idx] @ item_embeds_norm.T
        # Гибрид
        hybrid = alpha * als_vec + (1 - alpha) * cont_vec
        # Фильтруем просмотренные
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores