"""
train_xgb.py is the module for training a XGBClassifier pipeline
"""

# Libraries --------------------------------------------------------------------------------------------------------------------------

import argparse
import numpy as np
import os
import json
import logging
from pathlib import Path
import dask.dataframe as dask_df
from dask.distributed import LocalCluster, Client
import xgboost as xgb
from sklearn.metrics import (roc_curve, confusion_matrix, average_precision_score, f1_score, 
                            log_loss, precision_score, recall_score)

# Variables --------------------------------------------------------------------------------------------------------------------------

## Read environmental variables
def gcs_path_to_local_path(old_path):
    new_path = old_path.replace("gs://", "/gcs/")
    return new_path

TRAINING_DATA_PATH = gcs_path_to_local_path(os.environ["AIP_TRAINING_DATA_URI"])
TEST_DATA_PATH = gcs_path_to_local_path(os.environ["AIP_TEST_DATA_URI"])
MODEL_DIR = gcs_path_to_local_path(os.environ["AIP_MODEL_DIR"])
MODEL_PATH = MODEL_DIR + "model.bst"

## Training variables
LABEL_COLUMN = "tx_fraud"
UNUSED_COLUMNS = ["timestamp","entity_type_customer","entity_type_terminal"]
DATA_SCHEMA = {
"timestamp" : "object",
"tx_amount": "float64",
"tx_fraud": "Int64",
"entity_type_customer": "Int64",
"customer_id_nb_tx_1day_window": "Int64",
"customer_id_nb_tx_7day_window": "Int64",
"customer_id_nb_tx_14day_window": "Int64",
"customer_id_avg_amount_1day_window": "float64",
"customer_id_avg_amount_7day_window": "float64",
"customer_id_avg_amount_14day_window": "float64",
"customer_id_nb_tx_15min_window": "Int64",
"customer_id_avg_amount_15min_window": "float64",
"customer_id_nb_tx_30min_window": "Int64",
"customer_id_avg_amount_30min_window": "float64",
"customer_id_nb_tx_60min_window": "Int64",
"customer_id_avg_amount_60min_window": "float64",
"entity_type_terminal": "Int64",
"terminal_id_nb_tx_1day_window": "Int64",
"terminal_id_nb_tx_7day_window": "Int64",
"terminal_id_nb_tx_14day_window": "Int64",
"terminal_id_risk_1day_window": "float64",
"terminal_id_risk_7day_window": "float64",
"terminal_id_risk_14day_window": "float64",
"terminal_id_nb_tx_15min_window": "Int64",
"terminal_id_avg_amount_15min_window": "float64",
"terminal_id_nb_tx_30min_window": "Int64",
"terminal_id_avg_amount_30min_window": "float64",
"terminal_id_nb_tx_60min_window": "Int64",
"terminal_id_avg_amount_60min_window": "float64"
}

# Helpers -----------------------------------------------------------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    # Data files arguments
    parser.add_argument("--bucket", dest="bucket", type=str,
                        required=True, help="Bucket uri")
    parser.add_argument("--max_depth", dest="max_depth",
                        default=6, type=int,
                        help="max_depth value.")
    parser.add_argument("--eta", dest="eta",
                        default=0.4, type=float,
                        help="eta.")
    parser.add_argument("--gamma", dest="gamma",
                        default=0.0, type=float,
                        help="eta value")
    
    return parser.parse_args()

def resample(df, replace, frac=1, random_state = 8):
    shuffled_df = df.sample(frac=frac, replace=replace, random_state=random_state)
    return shuffled_df

def preprocess(df):
    df = df.drop(columns=UNUSED_COLUMNS)

    # Drop rows with NaN"s
    df = df.dropna()

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = df.select_dtypes(["float32", "float64"]).columns
    numeric_format = {col:"float32" for col in numeric_columns}
    df = df.astype(numeric_format)

    return df

def evaluate_model(model, x_true, y_true):
    y_true = y_true.compute()
    
    #calculate metrics
    metrics={}
    
    y_score =  model.predict_proba(x_true)[:, 1]
    y_score = y_score.compute()
    fpr, tpr, thr = roc_curve(
         y_true=y_true, y_score=y_score, pos_label=True
    )
    fpr_list = fpr.tolist()[::1000]
    tpr_list = tpr.tolist()[::1000]
    thr_list = thr.tolist()[::1000]

    y_pred = model.predict(x_true)
    y_pred.compute()
    c_matrix = confusion_matrix(y_true, y_pred)
    
    avg_precision_score = round(average_precision_score(y_true, y_score), 3)
    f1 = round(f1_score(y_true, y_pred), 3)
    lg_loss = round(log_loss(y_true, y_pred), 3)
    prec_score = round(precision_score(y_true, y_pred), 3)
    rec_score = round(recall_score(y_true, y_pred), 3)
    
    metrics["fpr"] = [round(f, 3) for f in fpr_list]
    metrics["tpr"] = [round(f, 3) for f in tpr_list]
    metrics["thrs"] = [round(f, 3) for f in thr_list]
    metrics["confusion_matrix"] = c_matrix.tolist()
    metrics["avg_precision_score"] = avg_precision_score
    metrics["f1_score"] = f1
    metrics["log_loss"] = lg_loss
    metrics["precision_score"] = prec_score
    metrics["recall_score"] = rec_score
    
    return metrics


def main():
    args = get_args()
        
    # variables
    bucket = gcs_path_to_local_path(args.bucket)
    deliverable_uri = (Path(bucket)/"deliverables")
    metrics_uri = (deliverable_uri/"metrics.json")

    # read data
    train_df = dask_df.read_csv(TRAINING_DATA_PATH, dtype=DATA_SCHEMA)
    test_df = dask_df.read_csv(TEST_DATA_PATH, dtype=DATA_SCHEMA)
    
    # preprocessing
    preprocessed_train_df = preprocess(train_df)
    preprocessed_test_df = preprocess(test_df)
    
    # downsampling
    train_nfraud_df = preprocessed_train_df[preprocessed_train_df[LABEL_COLUMN]==0]
    train_fraud_df = preprocessed_train_df[preprocessed_train_df[LABEL_COLUMN]==1]
    train_nfraud_downsample = resample(train_nfraud_df,
                          replace=True, 
                          frac=len(train_fraud_df)/len(train_df))
    ds_preprocessed_train_df = dask_df.multi.concat([train_nfraud_downsample, train_fraud_df])
    
    # target, features split
    x_train = ds_preprocessed_train_df[ds_preprocessed_train_df.columns.difference([LABEL_COLUMN])]
    y_train = ds_preprocessed_train_df.loc[:, LABEL_COLUMN].astype(int)
    x_true = preprocessed_test_df[preprocessed_test_df.columns.difference([LABEL_COLUMN])]
    y_true = preprocessed_test_df.loc[:, LABEL_COLUMN].astype(int)
    
    # train model
    cluster =  LocalCluster()
    client = Client(cluster)
    model = xgb.dask.DaskXGBClassifier(objective="reg:logistic", eval_metric="logloss")
    model.client = client 
    model.fit(x_train, y_train, eval_set=[(x_true, y_true)])
    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    
    #generate metrics
    metrics = evaluate_model(model, x_true, y_true)
    if not Path(deliverable_uri).exists():
        Path(deliverable_uri).mkdir(parents=True, exist_ok=True)
    with open(metrics_uri, "w") as file:
        json.dump(metrics, file, sort_keys = True, indent = 4)
    file.close()
    
if __name__ == "__main__":
    main()