"""
Script for serving.
"""
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request
from constants import SUBSCRIBER_FEATURES


OUTPUT_MODEL_NAME = "/artefact/train/xgb_model.pkl"


def predict_prob(subscriber_features,
                 model=pickle.load(open(OUTPUT_MODEL_NAME, "rb"))):
    """Predict subscribe probability given subscriber_features.
    Args:
        subscriber_features (dict)
        model
    Returns:
        sub_prob (float): subscribe probability
    """
    payloaddf = pd.DataFrame(subscriber_features,index=[0])
    xgpayload = xgb.DMatrix(payloaddf.values)
        
    # Score
    sub_prob = (
        model
        .predict(xgpayload)
        .item()
    )

    return sub_prob


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `sub_prob` given the subscriber features"""

    subscriber_features = request.json
    result = {
        "sub_prob": predict_prob(subscriber_features)
    }
    return result


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
