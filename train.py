
# import libraries
import logging
import os
import pickle

from bedrock_client.bedrock.api import BedrockApi
import numpy as np                                
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from helper import get_temp_data_bucket

TEMP_DATA_BUCKET = get_temp_data_bucket()
PREPROCESSED_DATA = TEMP_DATA_BUCKET + os.getenv("PREPROCESSED_DATA")

MAX_DEPTH = int(os.getenv("MAX_DEPTH"))
ETA = float(os.getenv("ETA"))
GAMMA = float(os.getenv("GAMMA"))
MIN_CHILD_WEIGHT = int(os.getenv("MIN_CHILD_WEIGHT"))
SUBSAMPLE = float(os.getenv("SUBSAMPLE"))
SILENT = int(os.getenv("SILENT"))
OBJECTIVE = os.getenv("OBJECTIVE")
NUM_ROUND = int(os.getenv("NUM_ROUND"))

def compute_log_metrics(bst, xgtest, test_data):
    """Compute and log metrics."""      
    print("\tEvaluating using validation data")
    preds = bst.predict(xgtest)
    y_target = test_data['y_yes']
    y_preds = np.round(preds).astype(int)
	
    acc = metrics.accuracy_score(y_target, y_preds)
    f1 = metrics.f1_score(y_target, y_preds)
    precision = metrics.precision_score(y_target, y_preds)
    recall = metrics.recall_score(y_target, y_preds)

    print("Accuracy = {:.6f}".format(acc))
    print("Precision = {:.6f}".format(precision))
    print("Recall = {:.6f}".format(recall))
    print("F1 score = {:.6f}".format(f1))

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 score", f1)
    bedrock.log_chart_data(y_target.astype(int).tolist(),
                           y_preds.flatten().tolist())

def main():
    """Train pipeline"""
    subscribers = pd.read_csv(PREPROCESSED_DATA)

    print("\tSplitting train and validation data")
    train_data, test_data = np.split(subscribers.sample(frac=1, random_state=1729), [int(0.7 * len(subscribers))])
    target = train_data['y_yes']
    train = train_data.drop(['y_yes'],axis=1)
    test = test_data.drop(['y_yes'],axis=1)
    xgtrain = xgb.DMatrix(train.values, target.values)
    xgtest = xgb.DMatrix(test.values)

    print("\tTrain model")
    param = {
	'max_depth':MAX_DEPTH,
	'eta':ETA,
	'gamma':GAMMA,
	'min_child_weight':MIN_CHILD_WEIGHT,
	'subsample':SUBSAMPLE,
	'silent':SILENT,
	'objective':OBJECTIVE
    }
    num_round=NUM_ROUND
    bst = xgb.train(param, xgtrain, num_round)

    compute_log_metrics(y_targets, y_preds)


    print("\tSaving model")
    os.mkdir("/artefact/train")
    with open("/artefact/train/" + OUTPUT_MODEL_NAME, "wb") as model_file:
        pickle.dump(bst, model_file)

if __name__ == "__main__":
    main()
