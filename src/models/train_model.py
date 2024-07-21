import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
from yaml import safe_load
import joblib

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


def main():
    root_path = Path(__file__).parent.parent.parent
    # load path
    train_data_path = root_path / "data" / "processed" / "train_processed.csv"
    # params_path
    params_path = root_path / "params.yaml"
    # save model path
    model_save_path = root_path / "models" 
    model_save_path.mkdir(exist_ok=True)
    
    # load the training and test data
    train_df = load_data(train_data_path)
    
    # split the train data into X_train and y_train
    X_train, y_train = make_X_y(train_df)
     
    # load the model parameters from params.yaml file
    model_params = read_params(params_path)['train_model']
    
    # make model pipeline
    clf = GradientBoostingClassifier(**model_params)
 
    # fit the model
    clf.fit(X_train,y_train)
    
    # save the fit model
    joblib.dump(clf,model_save_path / "model.joblib")
    
    
if __name__ == "__main__":
    main()