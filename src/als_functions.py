import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.gpu import HAS_CUDA

def create_csr(df, user_col='user_idx', item_col='item_idx', rating_col='watch_ratio'):
    """Создаёт CSR-матрицу из DataFrame."""
    rows = df[user_col].values
    cols = df[item_col].values
    data = df[rating_col].values.astype(np.float32)
    n_users = df[user_col].max() + 1
    n_items = df[item_col].max() + 1
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

def train_als(matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True):
    """Обучает ALS и возвращает модель."""
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=use_gpu and HAS_CUDA,
        random_state=42
    )
    model.fit(matrix)
    return model

def set_item_factors_from_embeddings(model, emb_matrix, item_count):
    """Заменяет факторы предметов на контентные эмбеддинги (для инициализации)."""
    # emb_matrix должен быть (n_items, factors) и совпадать по порядку с индексами
    if emb_matrix.shape[1] != model.factors:
        # Обрезаем или дополняем до нужной размерности
        if emb_matrix.shape[1] < model.factors:
            # Дополняем нулями
            pad = model.factors - emb_matrix.shape[1]
            emb = np.hstack([emb_matrix, np.zeros((emb_matrix.shape[0], pad))])
        else:
            emb = emb_matrix[:, :model.factors]
    else:
        emb = emb_matrix
    # Приводим к типу float32
    model.item_factors = emb.astype(np.float32)
    return model

def recommend(model, user_ids, item_ids, N=10):
    """Возвращает топ-N рекомендаций для списка пользователей."""
    return model.recommend(user_ids, item_ids, N=N)