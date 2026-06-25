import pandas as pd
import numpy as np

def add_watch_ratio(df_train, items_meta):
    """
    Добавляем колонку watch_ratio = timespent / duration, зажатую в [0,1].
    Предполагается, что в df_train есть 'item_id' и 'timespent'.
    """
    # Объединяем с метаданными
    df = df_train.merge(items_meta[['item_id', 'duration']], on='item_id', how='left')
    # Защита от деления на ноль
    duration = df['duration'].clip(lower=1)
    df['watch_ratio'] = (df['timespent'] / duration).clip(0, 1)
    return df