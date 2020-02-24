version = "1.0"

train {
    step preprocess {
        image = "basisai/workload-standard"
        install = ["pip3 install -r requirements.txt"]
        script = [
            {
                sh = ["python3 preprocess.py"]
            }
        ]
        resources {
            cpu = "500m"
            memory = "500M"
        }
    }
    
    step train {
        image = "basisai/workload-standard"
        install = ["pip3 install -r requirements.txt"]
        script = [
            {
                sh = ["python3 train.py"]
            }
        ]
        resources {
            cpu = "500m"
            memory = "500M"
        }
        depends_on = ["preprocess"]
    }
    
    parameters {
        RAW_SUBSCRIBERS_DATA = "https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv"
        TEMP_DATA_BUCKET = "gs://span-temp-production/"
        PREPROCESSED_DATA = "subscriber_data/preprocessed"
      	MAX_DEPTH = "5"
      	ETA = "0.2"
      	GAMMA = "4"
      	MIN_CHILD_WEIGHT = "6"
      	SUBSAMPLE = "0.8"
      	SILENT = "0"
      	OBJECTIVE = "binary:logistic"
      	NUM_ROUND = "100"
        OUTPUT_MODEL_NAME = "xgb_model.pkl"
    }
    
    #secrets = [
    #    "DUMMY_SECRET_A",
    #    "DUMMY_SECRET_B"
    #]
}


serve {
    image = "python:3.7"
    install = ["pip3 install -r requirements.txt"]
    script = [{sh = ["gunicorn --bind=:${SERVER_PORT} --worker-class=gthread --timeout=300 serve_http:app"]}]
}
