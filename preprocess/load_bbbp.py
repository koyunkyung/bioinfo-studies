import pandas as pd
import numpy as np
from graphs import *

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df = pd.read_csv(url, usecols=[1, 2, 3])
df.to_csv('data/BBBP.csv')

permuted_indices = np.random.permutation(np.arange(df.shape[0]))

train_index = permuted_indices[: int(df.shape[0] * 0.8)]
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
test_index = permuted_indices[int(df.shape[0] * 0.99) :]

train_df = df.iloc[train_index]
valid_df = df.iloc[valid_index]
test_df = df.iloc[test_index]

train_df.to_csv('data/train.csv', index=False)
valid_df.to_csv('data/valid.csv', index=False)
test_df.to_csv('data/test.csv', index=False)