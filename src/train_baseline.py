import pickle
from get_data import load_data
from new_basic_feature import add_watch_ratio
from als_functions import create_csr, train_als

df_train, df_val, items_meta, emb_matrix, user_map, item_map = load_data("up0.01_ir0.01")
df_train = add_watch_ratio(df_train, items_meta)
df_val = add_watch_ratio(df_val, items_meta)
train_matrix = create_csr(df_train)

print("Training baseline ALS")
model = train_als(train_matrix, factors=64, regularization=0.1, iterations=15, use_gpu=True)

with open("baseline_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'train_matrix': train_matrix,
        'df_val': df_val,
        'emb_matrix': emb_matrix,
        'user_map': user_map,
        'item_map': item_map,
        'items_meta': items_meta
    }, f)
print("Model saved to baseline_model.pkl")