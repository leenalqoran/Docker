import pandas as pd

def load_data(train_path, valid_path):
    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    return train_data, valid_data
