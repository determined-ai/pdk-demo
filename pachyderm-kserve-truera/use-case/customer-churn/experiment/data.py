import os
import shutil

import python_pachyderm
from python_pachyderm.proto.v2.pfs.pfs_pb2 import FileType

import pandas as pd
import numpy as np

from utils import *

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

class Churn_Dataset(Dataset):
 
  def __init__(self, df, training_cols, label_col):
 
    self.X = torch.tensor(df[training_cols].values, dtype=torch.float32)
    self.y = torch.tensor(df[label_col].values, dtype=torch.float32).unsqueeze(-1)
 
  def __len__(self):
    return len(self.y)
  
  def __getitem__(self,idx):
    return self.X[idx], self.y[idx]

def get_train_and_validation_datasets(files, test_size=0.2, random_seed=42):
    
    # Read csv files one by one and concatenate them to get the full dataset
    full_df = pd.read_csv(files[0])
    
    for file in files[1:]:
        partial_df = pd.read_csv(file)
        full_df = pd.concat([full_df, partial_df], axis=0)
        full_df.reset_index(drop=True, inplace=True)
        
    train_df, val_df = train_test_split(full_df, test_size=test_size, random_state=random_seed)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    object_cols = list(train_df.columns[train_df.dtypes.values == "object"])
    int_cols = list(train_df.columns[train_df.dtypes.values == "int"])
    float_cols = list(train_df.columns[train_df.dtypes.values == "float"])

    # Churn will be the label, no need to preprocess it
    int_cols.remove("churn")

    numerical_cols = int_cols+float_cols
    
    # Keep an unscaled version of train_df for scaling all dataframes
    unscaled_train_df = train_df.copy()

    train_df = preprocess_dataframe(train_df, unscaled_train_df, numerical_cols)
    val_df = preprocess_dataframe(val_df, unscaled_train_df, numerical_cols)
    
    training_cols = list(train_df.columns)
    label_col = "churn"
    training_cols.remove(label_col)
    
    train_dataset = Churn_Dataset(train_df, training_cols, label_col)
    val_dataset = Churn_Dataset(val_df, training_cols, label_col)
    
    return train_dataset, val_dataset


def download_pach_repo(pachyderm_host, pachyderm_port, repo, branch, root, token):
    print(f"Starting to download dataset: {repo}@{branch} --> {root}")

    if not os.path.exists(root):
        os.makedirs(root)

    client = python_pachyderm.Client(host=pachyderm_host, port=pachyderm_port, auth_token=token)
    files = []

    for diff in client.diff_file((repo, branch), "/"):
        src_path = diff.new_file.file.path
        des_path = os.path.join(root, src_path[1:])
        # print(f"Got src='{src_path}', des='{des_path}'")

        if diff.new_file.file_type == FileType.FILE:
            if src_path != "":
                files.append((src_path, des_path))
        elif diff.new_file.file_type == FileType.DIR:
            print(f"Creating dir : {des_path}")
            os.makedirs(des_path, exist_ok=True)

    for src_path, des_path in files:
        src_file = client.get_file((repo, branch), src_path)
        # print(f'Downloading {src_path} to {des_path}')

        with open(des_path, "wb") as dest_file:
            shutil.copyfileobj(src_file, dest_file)

    print("Download operation ended")
    return files