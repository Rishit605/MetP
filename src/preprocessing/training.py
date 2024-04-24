import pandas as pd
import numpy as np

from data_pipeline import pre_process
from sklearn.preprocessing import MinMaxScaler

def scaler():
  scale = MinMaxScaler(feature_range=(0,1))

  _X = window_make()


def window_make(df):
  df_as_np = df.to_numpy()


  X=[]
  y=[]

  for i in range(len(df_as_np)-window_size):
    row = [a for a in df_as_np[i : i + window_size]]
    X.append(row)

    label = [df_as_np[i + window_size][0],
             df_as_np[i + window_size][1],
             df_as_np[i + window_size][2],
             df_as_np[i + window_size][3],
             df_as_np[i + window_size][4]]
    y.append(label)

  return np.array(X), np.array(y)


def ret_split_data(data: pd.DataFrame):
    X1, y1 = window_make(data)

    train_split = int(len(X1) * (70 / 100))

    if len(data) < 10000: 
        valid_split = int(len(X1) - train_split)

        # Splitting the Dataset
        X1_train, y1_train = X1[:train_split], y1[:train_split]
        X1_val, y1_val = X1[train_split:valid_split], y1[train_split:valid_split]
        return X1_train, y1_train, X1_val, y1_val


    else:
        valid_split = int(len(X1) - train_split)
        test_split = int(valid_split - (50 / 100))

        # Splitting the Dataset
        X1_train, y1_train = X1[:train_split], y1[:train_split]
        X1_val, y1_val = X1[train_split:valid_split], y1[train_split:valid_split]
        X1_test, y1_test = X1[valid_split:test_split], y1[valid_split:test_split]
        return X1_train, y1_train, X1_val, y1_val, X1_test, y1_test
    

res = ret_split_data(pre_process())

if isinstance(res, tuple):
  if len(result) == 4:
    X_Train, y_Train, X_Test, y_Test = ret_split_data(pre_process())
  elif len(result) == 6:
    X_Train, y_Train, X_Val, y_Val, X_Test, y_Test = ret_split_data(pre_process())


