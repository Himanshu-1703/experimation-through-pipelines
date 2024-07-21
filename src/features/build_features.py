import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from pathlib import Path
from yaml import safe_load
from sklearn import set_config
import joblib

set_config(transform_output="pandas")

TARGET = "Placed"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def read_params(file_path: Path) -> dict:
    with open(file_path,'r') as file:
        params = safe_load(file)
    return params


def make_X_y(df: pd.DataFrame, target_column: str = TARGET):
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y

def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,save_path: Path) -> None:
    # save the training data
    train_data.to_csv(save_path / "train_processed.csv",index=False)
    # save the test data
    test_data.to_csv(save_path / "test_processed.csv",index=False)
    
    
def main():
    root_path = Path(__file__).parent.parent.parent
    # load path
    data_path = root_path / "data" / "raw" 
    # params_path
    params_path = root_path / "params.yaml"
    # save path
    save_path = data_path.parent / "processed"
    save_path.mkdir(exist_ok=True)
    # save preprocessor path
    preprocessor_save_path = root_path / "models" 
    preprocessor_save_path.mkdir(exist_ok=True)
    
    # load the training and test data
    train_df = load_data(data_path / "train.csv")
    test_df = load_data(data_path / "test.csv")
    
    # split the data into X and y
    X_train, y_train = make_X_y(train_df)
    X_test, y_test = make_X_y(test_df)
    
    # read the parameters from params file
    n_components = read_params(params_path)['build_features']['n_components']
    
    # make the preprocessor object
    preprocessor = Pipeline(steps=[
        ('scaler',StandardScaler()),
        ('pca',PCA(n_components=n_components))
    ])
    
    # fit and transform the X_train
    X_train_trans = preprocessor.fit_transform(X_train)
    
    # transform the test data\
    X_test_trans = preprocessor.transform(X_test)
    
    # combine the data
    X_train_trans[TARGET] = y_train
    X_test_trans[TARGET] = y_test
    
    # save the data
    save_data(X_train_trans,
              X_test_trans,
              save_path)
    
    # save preprocessor
    joblib.dump(preprocessor,preprocessor_save_path / "preprocessor.joblib")
    
    
    
if __name__ == "__main__":
    main()