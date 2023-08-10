# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib

import random
import string

import datetime as dt
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import List, Union


from kfp.v2 import compiler, dsl
from pipelines import generate_query
from bigquery_components import (
    ingest_features_gcs,
    evaluate_model,
    copy_bigquery_data_comp,
    extract_bq_to_dataset,
    feature_engineering_comp,
)
from vertex_components import (
    lookup_model,
    custom_train_job,
    import_model_evaluation,
    update_best_model,
)
from google_cloud_pipeline_components import aiplatform as vertex_ai_components


@dsl.pipeline(name="xgboost-train-pipeline")
def xgboost_pipeline(
    project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    project_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_project_id: str = os.environ.get("VERTEX_PROJECT_ID"),
    model_name: str = "simple_xgboost",
    dataset_id: str = "preprocessing",
    dataset_location: str = os.environ.get("VERTEX_LOCATION"),
    ingestion_dataset_id: str = "chicago_taxi_trips",
    timestamp: str = "2022-12-01 00:00:00",
    staging_bucket: str = os.environ.get("VERTEX_PIPELINE_ROOT"),
    pipeline_files_gcs_path: str = os.environ.get("PIPELINE_FILES_GCS_PATH"),
    test_dataset_uri: str = "",
):
    """
    XGB training pipeline which:
     1. Splits and extracts a dataset from BQ to GCS
     2. Trains a model via Vertex AI CustomTrainingJob
     3. Evaluates the model against the current champion model
     4. If better the model becomes the new default model

    Args:
        project_id (str): project id of the Google Cloud project
        project_location (str): location of the Google Cloud project
        ingestion_project_id (str): project id containing the source bigquery data
            for ingestion. This can be the same as `project_id` if the source data is
            in the same project where the ML pipeline is executed.
        model_name (str): name of model
        dataset_id (str): id of BQ dataset used to store all staging data & predictions
        dataset_location (str): location of dataset
        ingestion_dataset_id (str): dataset id of ingestion data
        timestamp (str): Optional. Empty or a specific timestamp in ISO 8601 format
            (YYYY-MM-DDThh:mm:ss.sss±hh:mm or YYYY-MM-DDThh:mm:ss).
            If any time part is missing, it will be regarded as zero.
        staging_bucket (str): Staging bucket for pipeline artifacts.
        pipeline_files_gcs_path (str): GCS path where the pipeline files are located.
        test_dataset_uri (str): Optional. GCS URI of statis held-out test dataset.
    """

    # Create variables to ensure the same arguments are passed
    # into different components of the pipeline
    label_column_name = "total_fare"
    time_column = "trip_start_timestamp"
    ingestion_table = "taxi_trips"
    table_suffix = "_xgb_training"  # suffix to table names
    ingested_table = "ingested_data" + table_suffix
    preprocessed_table = "preprocessed_data" + table_suffix
    train_table = "train_data" + table_suffix
    valid_table = "valid_data" + table_suffix
    test_table = "test_data" + table_suffix
    primary_metric = "rootMeanSquaredError"
    train_script_uri = f"{pipeline_files_gcs_path}/training/assets/train_xgb_model.py"
    hparams = dict(
        n_estimators=200,
        early_stopping_rounds=10,
        objective="reg:squarederror",
        booster="gbtree",
        learning_rate=0.3,
        min_split_loss=0,
        max_depth=6,
        label=label_column_name,
    )

    ID = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    FEATURESTORE_ID = "fraudfinder_{ID}"
    YESTERDAY = datetime.today() - timedelta(days=1)
    YEAR_MONTH_PREFIX = YESTERDAY.strftime("%Y-%m")
    DATAPROCESSING_START_DATE = (YESTERDAY - timedelta(days=10)).strftime("%Y-%m-%d")
    DATAPROCESSING_END_DATE = YESTERDAY.strftime("%Y-%m-%d")

    # Define BiqQuery dataset and tables to calculate features.
    RAW_BQ_TRANSACTION_TABLE_URI = f"{project_id}.tx.tx"
    RAW_BQ_LABELS_TABLE_URI = f"{project_id}.tx.txlabels"
    FEATURES_BQ_TABLE_URI = f"{project_id}.tx.wide_features_table"

    # Define Vertex AI Feature store settings.
    CUSTOMERS_TABLE_NAME = f"customers_{DATAPROCESSING_END_DATE.replace('-', '')}"
    CUSTOMERS_BQ_TABLE_URI = f"{project_id}.tx.{CUSTOMERS_TABLE_NAME}"
    TERMINALS_TABLE_NAME = f"terminals_{DATAPROCESSING_END_DATE.replace('-', '')}"
    TERMINALS_BQ_TABLE_URI = f"{project_id}.tx.{TERMINALS_TABLE_NAME}"
    ONLINE_STORAGE_NODES = 1
    FEATURE_TIME = "feature_ts"
    CUSTOMER_ENTITY_ID = "customer"
    TERMINAL_ENTITY_ID = "terminal"

    # Feature Store component variables
    BQ_DATASET = "tx"
    READ_INSTANCES_TABLE = f"ground_truth_{ID}"
    READ_INSTANCES_URI = f"bq://{project_id}.{BQ_DATASET}.{READ_INSTANCES_TABLE}"

    # Dataset component variables
    DATASET_NAME = f"fraud_finder_dataset_{ID}"

    # Training component variables
    JOB_NAME = f"fraudfinder-train-xgb-{ID}"
    MODEL_NAME = f"ff_model_xgb_pipeline_{ID}"
    CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-1:latest"
    MODEL_SERVING_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest"
    )
    ARGS = json.dumps(["--bucket", f"gs://{staging_bucket}"])
    IMAGE_REPOSITORY = f"fraudfinder-{project_id}"
    IMAGE_NAME = "dask-xgb-classificator"
    IMAGE_TAG = "latest"
    IMAGE_URI = f"us-central1-docker.pkg.dev/{project_id}/{IMAGE_REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"  # TODO: get it from config

    # Evaluation component variables
    METRICS_URI = f"gs://{staging_bucket}/deliverables/metrics.json"
    AVG_PR_THRESHOLD = 0.2
    AVG_PR_CONDITION = "avg_pr_condition"

    replica_count = 1
    machine_type = "n1-standard-4"
    train_split = 0.8
    test_split = 0.1
    val_split = 0.1

    # generate sql queries which are used in ingestion and preprocessing
    # operations

    queries_folder = pathlib.Path(__file__).parent / "queries"

    # data ingestion and preprocessing operations

    # kwargs = dict(
    #     bq_client_project_id=project_id,
    #     destination_project_id=project_id,
    #     dataset_id=dataset_id,
    #     dataset_location=dataset_location,
    #     query_job_config=json.dumps(dict(write_disposition="WRITE_TRUNCATE"))
    # )

    ingest = copy_bigquery_data_comp(
        bucket_name=staging_bucket, destination_project_id=project_id
    ).set_display_name("Ingest data")

    import logging
    logging.getLogger().setLevel(logging.INFO)
    logmsg = f"Project location = {project_location}"

    logging.info(logmsg)

    # featurestore = (
    #     feature_engineering_comp(
    #         destination_project_id=project_id,
    #         REGION_NM=project_location,
    #         BUCKET_NAME=staging_bucket,
    #         FEATURESTORE_ID=FEATURESTORE_ID,
    #         DATAPROCESSING_END_DATE=DATAPROCESSING_END_DATE,
    #         RAW_BQ_TRANSACTION_TABLE_URI=RAW_BQ_TRANSACTION_TABLE_URI,
    #         RAW_BQ_LABELS_TABLE_URI=RAW_BQ_LABELS_TABLE_URI,
    #         CUSTOMERS_BQ_TABLE_URI=CUSTOMERS_BQ_TABLE_URI,
    #         TERMINALS_BQ_TABLE_URI=TERMINALS_BQ_TABLE_URI,
    #         ONLINE_STORAGE_NODES=ONLINE_STORAGE_NODES,
    #     )
    #     .after(ingest)
    #     .set_display_name("Create Feature Store")
    # )

    # Ingest data from featurestore
    ingest_features_op = ingest_features_gcs(
        project_id=project_id,
        region=project_location,
        bucket_name=staging_bucket,
        feature_store_id=FEATURESTORE_ID,
        read_instances_uri=READ_INSTANCES_URI,
    )

    # create dataset
    dataset_create_op = vertex_ai_components.TabularDatasetCreateOp(
        project=project_id,
        display_name=DATASET_NAME,
        gcs_source=ingest_features_op.outputs["snapshot_uri_paths"],
    ).after(ingest_features_op)

    # custom training job component - script
    train_model_op = vertex_ai_components.CustomContainerTrainingJobRunOp(
        display_name=JOB_NAME,
        model_display_name=MODEL_NAME,
        container_uri=IMAGE_URI,
        staging_bucket=staging_bucket,
        dataset=dataset_create_op.outputs["dataset"],
        base_output_dir=staging_bucket,
        args=ARGS,
        replica_count=replica_count,
        machine_type=machine_type,
        training_fraction_split=train_split,
        validation_fraction_split=val_split,
        test_fraction_split=test_split,
        model_serving_container_image_uri=MODEL_SERVING_IMAGE_URI,
        project=project_id,
        location=project_location,
    ).after(dataset_create_op)

    # evaluate component
    evaluate_model_op = evaluate_model(
        model_in=train_model_op.outputs["model"], metrics_uri=METRICS_URI
    ).after(train_model_op)

    # split_train_data = (
    #     bq_query_to_table(query=split_train_query, table_id=train_table, **kwargs)
    #     .after(ingest)
    #     .set_display_name("Split train data")
    # )
    # split_valid_data = (
    #     bq_query_to_table(query=split_valid_query, table_id=valid_table, **kwargs)
    #     .after(ingest)
    #     .set_display_name("Split validation data")
    # )
    # split_test_data = (
    #     bq_query_to_table(query=split_test_query, table_id=test_table, **kwargs)
    #     .after(ingest)
    #     .set_display_name("Split test data")
    # )
    # data_cleaning = (
    #     bq_query_to_table(
    #         query=data_cleaning_query, table_id=preprocessed_table, **kwargs
    #     )
    #     .after(split_train_data)
    #     .set_display_name("Clean data")
    # )

    # # data extraction to gcs

    # train_dataset = (
    #     extract_bq_to_dataset(
    #         bq_client_project_id=project_id,
    #         source_project_id=project_id,
    #         dataset_id=dataset_id,
    #         table_name=preprocessed_table,
    #         dataset_location=dataset_location,
    #     )
    #     .after(data_cleaning)
    #     .set_display_name("Extract train data to storage")
    # ).outputs["dataset"]
    # valid_dataset = (
    #     extract_bq_to_dataset(
    #         bq_client_project_id=project_id,
    #         source_project_id=project_id,
    #         dataset_id=dataset_id,
    #         table_name=valid_table,
    #         dataset_location=dataset_location,
    #     )
    #     .after(split_valid_data)
    #     .set_display_name("Extract validation data to storage")
    # ).outputs["dataset"]
    # test_dataset = (
    #     extract_bq_to_dataset(
    #         bq_client_project_id=project_id,
    #         source_project_id=project_id,
    #         dataset_id=dataset_id,
    #         table_name=test_table,
    #         dataset_location=dataset_location,
    #         destination_gcs_uri=test_dataset_uri,
    #     )
    #     .after(split_test_data)
    #     .set_display_name("Extract test data to storage")
    #     .set_caching_options(False)
    # ).outputs["dataset"]

    # existing_model = (
    #     lookup_model(
    #         model_name=model_name,
    #         project_location=project_location,
    #         project_id=project_id,
    #         fail_on_model_not_found=False,
    #     )
    #     .set_display_name("Lookup past model")
    #     .set_caching_options(False)
    #     .outputs["model_resource_name"]
    # )

    # train_model = custom_train_job(
    #     train_script_uri=train_script_uri,
    #     train_data=train_dataset,
    #     valid_data=valid_dataset,
    #     test_data=test_dataset,
    #     project_id=project_id,
    #     project_location=project_location,
    #     model_display_name=model_name,
    #     train_container_uri="europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-23:latest",  # noqa: E501
    #     serving_container_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",  # noqa: E501
    #     hparams=hparams,
    #     requirements=["scikit-learn==0.24.0"],
    #     staging_bucket=staging_bucket,
    #     parent_model=existing_model,
    # ).set_display_name("Train model")

    # evaluation = import_model_evaluation(
    #     model=train_model.outputs["model"],
    #     metrics=train_model.outputs["metrics"],
    #     test_dataset=test_dataset,
    #     pipeline_job_id="{{$.pipeline_job_name}}",
    #     project_location=project_location,
    # ).set_display_name("Import evaluation")

    # with dsl.Condition(existing_model != "", "champion-exists"):
    #     update_best_model(
    #         challenger=train_model.outputs["model"],
    #         challenger_evaluation=evaluation.outputs["model_evaluation"],
    #         parent_model=existing_model,
    #         eval_metric=primary_metric,
    #         eval_lower_is_better=True,
    #         project_id=project_id,
    #         project_location=project_location,
    #     ).set_display_name("Update best model")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=xgboost_pipeline,
        package_path="training.json",
        type_check=False,
    )
