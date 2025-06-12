import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    df = pd.read_csv(file_path, header=None, names=columns)
    return df