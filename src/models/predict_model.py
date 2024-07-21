import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)


TARGET = "Placed"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def make_X_y(df: pd.DataFrame, target_column: str = TARGET):
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y


def main():
    root_path = Path(__file__).parent.parent.parent
    # load path
    test_data_path = root_path / "data" / "processed" / "test_processed.csv"
    # save model path
    model_load_path = root_path / "models" / "model.joblib"
    # plot save location
    save_plot_path = root_path / "reports" / "figures"
    save_plot_path.mkdir(exist_ok=True)
    # metrics save location
    metrics_save_path = root_path / "reports"
    metrics_save_path.mkdir(exist_ok=True)
    
    
    # load the training and test data
    test_df = load_data(test_data_path)
    
    # split the train data into X_train and y_train
    X_test, y_test = make_X_y(test_df)
    
    # load model
    clf = joblib.load(model_load_path)
    
    # get model predictions
    y_pred = clf.predict(X_test)
    
    # calculate the metrics
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    
    # save the metrics as json
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    
    with open(metrics_save_path / "metrics.json","w") as file:
        json.dump(metrics_dict,file,indent=4)
        
    # confusion matrix plot
    cm = (
        ConfusionMatrixDisplay
        .from_estimator(clf,X_test,y_test)
        .figure_
        )
    
    # save the figure
    cm.savefig(save_plot_path / "confusion_matrix.png")
    
    
    
if __name__ == "__main__":
    main()