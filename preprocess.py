import os
import pandas as pd
from helper import get_temp_data_bucket

RAW_SUBSCRIBERS_DATA = os.getenv("RAW_SUBSCRIBERS_DATA")
TEMP_DATA_BUCKET = get_temp_data_bucket()
PREPROCESSED_DATA = TEMP_DATA_BUCKET + os.getenv("PREPROCESSED_DATA")

def main():
    """Preprocess data"""
    print("\tPreprocessing")
    subscribers = pd.read_csv(RAW_SUBSCRIBERS_DATA)
	subscribers = subscribers.drop(['y_no'],axis=1)
	subscribers.to_csv(PREPROCESSED_DATA,index=false)

if __name__ == "__main__":
    main()