from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sklearn
import boto3
import pathlib
from io import StringIO
import argparse
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client and are passed as command line arguments to the script.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, Model and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args, _ = parser.parse_known_args()
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading Data")
    print()

    # Creating the Training and Testing dataset by reading the files using the args.train and args.test values using args.train_file and args.test_file
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    print("Building the training and testing dataset")

    features = list(train_df.columns)
    label = features.pop(-1)

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print("Column order :")
    print(features)
    print()

    print("Label Column is: ", label)
    print()

    print("Data Shape : ")
    print()

    print("-----Shape of the Training Data----")
    print(X_train.shape)
    print(y_train.shape)
    print()

    print("-----Shape of the Testing  Data----")
    print(X_test.shape)
    print(y_test.shape)
    print()

    print("Training RandomForest Model")
    print()

    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    # model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print()

    # dumping the model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at" + model_path)
    print()

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print()
    print("-----Metrics Results for Testing Data----")
    print()

    print("Total Rows are: ", X_test.shape[0])
    print("[Testing] Model accuracy is: ", test_acc)
    print("[Testing] Testing Report is: ")
    print(test_rep)
