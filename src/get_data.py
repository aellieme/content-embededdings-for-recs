import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from glob import glob
import os

REPO = "deepvk/VK-LSVD"

def load_data(subset="up0.01_ir0.01"):
    # Загрузка тренировочных 
    train_files = []
    for week in range(0, 25):
        fname = f"subsamples/{subset}/train/week_{week:02d}.parquet"
        try:
            local = hf_hub_download(REPO, fname, repo_type="dataset")
            train_files.append(local)
        except:
            pass  # некоторые подвыборки могут не содержать все недели
    df_train = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)

    # Валидация
    val_file = f"subsamples/{subset}/validation/week_25.parquet"
    local_val = hf_hub_download(REPO, val_file, repo_type="dataset")
    df_val = pd.read_parquet(local_val)

    # Метаданные предметов
    local_items = hf_hub_download(REPO, "metadata/items_metadata.parquet", repo_type="dataset")
    items_meta = pd.read_parquet(local_items)

    # Эмбеддинги
    local_emb = hf_hub_download(REPO, "metadata/item_embeddings.npz", repo_type="dataset")
    emb_data = np.load(local_emb)
    item_embeddings = pd.DataFrame({
        'item_id': emb_data['item_id'],
        'embedding': list(emb_data['embedding'])
    })

    # Маппинг ID -> индекс
    users = pd.concat([df_train['user_id'], df_val['user_id']]).unique()
    items = pd.concat([df_train['item_id'], df_val['item_id']]).unique()
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: j for j, i in enumerate(items)}

    # Преобразование
    for df in [df_train, df_val]:
        df['user_idx'] = df['user_id'].map(user_map)
        df['item_idx'] = df['item_id'].map(item_map)

    # Фильтруем эмбеддинги только для присутствующих item_id
    emb_filtered = item_embeddings[item_embeddings['item_id'].isin(items)]
    # Сортируем по item_idx
    emb_filtered['item_idx'] = emb_filtered['item_id'].map(item_map)
    emb_filtered = emb_filtered.sort_values('item_idx')
    emb_matrix = np.vstack(emb_filtered['embedding'].values).astype(np.float32)

    return df_train, df_val, items_meta, emb_matrix, user_map, item_map