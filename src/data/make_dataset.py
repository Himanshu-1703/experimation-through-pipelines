import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
from yaml import safe_load


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_data(df: pd.DataFrame,
               test_size: float,
               random_state: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    
    train_df, test_df = train_test_split(df,
                                         test_size=test_size,
                                         random_state=random_state)
    return train_df, test_df


def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,save_path: Path) -> None:
    # save the training data
    train_data.to_csv(save_path / "train.csv",index=False)
    # save the test data
    test_data.to_csv(save_path / "test.csv",index=False)
    
def read_params(file_path: Path) -> dict:
    with open(file_path,'r') as file:
        params = safe_load(file)
    return params

def main():
    root_path = Path(__file__).parent.parent.parent
    # load path
    data_path = root_path / "data" / "ingested" / "student_performance.csv"
    # params_path
    params_path = root_path / "params.yaml"
    # save path
    save_path = data_path.parent.parent / "raw"
    save_path.mkdir(exist_ok=True)
    # load the data
    df = load_data(data_path)
    # read the parameters
    params = read_params(params_path)['make_dataset']
    # split the data into train and test
    train_df, test_df = split_data(df=df,
                                   test_size=params['test_size'],
                                   random_state=params['random_state'])
    # save the data
    save_data(train_df,test_df,save_path)
    
if __name__ == "__main__":
    main()