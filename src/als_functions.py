import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA
from sklearn.metrics.pairwise import cosine_similarity

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

def compute_content_scores(user_embeddings, item_embeddings):
    return cosine_similarity(user_embeddings, item_embeddings)

def hybrid_predict(model, user_ids, train_matrix, alpha, content_scores, N=10):
    """
    Гибридный предсказатель: alpha * ALS + (1-alpha) * content,
    с фильтрацией уже просмотренных в train_matrix.
    """
    hybrid_scores = []
    for user_idx in user_ids:
        als_vec = model.user_factors[user_idx] @ model.item_factors.T
        cont_vec = content_scores[user_idx]
        hybrid = alpha * als_vec + (1 - alpha) * cont_vec
        # Фильтруем предметы, которые пользователь уже видел в обучении
        seen = train_matrix[user_idx].indices
        hybrid[seen] = -np.inf
        top = np.argsort(hybrid)[::-1][:N]
        hybrid_scores.append(top)
    return hybrid_scores