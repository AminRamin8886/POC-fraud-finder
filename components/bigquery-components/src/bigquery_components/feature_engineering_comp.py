# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp.v2.dsl import component
from typing import Union
import pandas as pd
from google.cloud import aiplatform as vertex_ai
from google.cloud import bigquery
from google.cloud.aiplatform import EntityType, Feature, Featurestore

# Define the date range of transactions for feature engineering (last 10 days up until yesterday)


@component(
    base_image="python:3.7",
    packages_to_install=["google-cloud-bigquery==2.30.0"],
)
def feature_engineering_comp(         
    destination_project_id: str,
    REGION: str,
    BUCKET_NAME: str,
    FEATURESTORE_ID: str,
    DATAPROCESSING_END_DATE: str ,    
    RAW_BQ_TRANSACTION_TABLE_URI: str,
    RAW_BQ_LABELS_TABLE_URI: str ,    
    CUSTOMERS_BQ_TABLE_URI: str ,
    TERMINALS_BQ_TABLE_URI: str ,
    ONLINE_STORAGE_NODES : int,
    FEATURE_TIME = "feature_ts",
    CUSTOMER_ENTITY_ID = "customer",
    TERMINAL_ENTITY_ID = "terminal"

) -> None:
    """
    Run query & create a new BigQuery table
    Args:    
        destination_project_id (str): project id where BQ table will be created        

    Returns:
        None
    """
    from google.cloud.exceptions import GoogleCloudError
    
    import logging

    logging.getLogger().setLevel(logging.INFO)

    
    try:
        create_customer_feature_table(DATAPROCESSING_END_DATE ,
        RAW_BQ_TRANSACTION_TABLE_URI,
        RAW_BQ_LABELS_TABLE_URI ,
        CUSTOMERS_BQ_TABLE_URI)
        logging.info(f"Done create_customer_feature_table")

        create_terminal_feature_table(DATAPROCESSING_END_DATE ,
        RAW_BQ_TRANSACTION_TABLE_URI,
        RAW_BQ_LABELS_TABLE_URI ,
        TERMINALS_BQ_TABLE_URI )
        logging.info(f"Done create_terminal_feature_table")

        create_real_time_features(CUSTOMERS_BQ_TABLE_URI, TERMINALS_BQ_TABLE_URI)
        logging.info(f"Done create_real_time_features")

        create_featurestore(destination_project_id, REGION, BUCKET_NAME,
        FEATURESTORE_ID, ONLINE_STORAGE_NODES, CUSTOMER_ENTITY_ID, TERMINAL_ENTITY_ID,
        CUSTOMERS_BQ_TABLE_URI, FEATURE_TIME, TERMINALS_BQ_TABLE_URI)
        logging.info(f"Done create_featurestore")

        logging.info(f"Done feature engineering")
    except GoogleCloudError as e:
        logging.error(e)        
        raise e


def run_bq_query(sql: str) -> Union[str, pd.DataFrame]:
    """
    Input: SQL query, as a string, to execute in BigQuery
    Returns the query results as a pandas DataFrame, or error, if any
    """
    
    from google.cloud import bigquery
    
    bq_client = bigquery.Client()
    
    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    bq_client.query(sql, job_config=job_config)

    # If dry run succeeds without errors, proceed to run query
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql, job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    
    return df

 
def create_customer_feature_table(   
    DATAPROCESSING_END_DATE ,
    RAW_BQ_TRANSACTION_TABLE_URI,
    RAW_BQ_LABELS_TABLE_URI ,
    CUSTOMERS_BQ_TABLE_URI
):
   

    create_customer_batch_features_query = f"""
    CREATE OR REPLACE TABLE `{CUSTOMERS_BQ_TABLE_URI}` AS
    WITH
    -- query to join labels with features -------------------------------------------------------------------------------------------
    get_raw_table AS (
    SELECT
        raw_tx.TX_TS,
        raw_tx.TX_ID,
        raw_tx.CUSTOMER_ID,
        raw_tx.TERMINAL_ID,
        raw_tx.TX_AMOUNT,
        raw_lb.TX_FRAUD
    FROM (
        SELECT
        *
        FROM
        `{RAW_BQ_TRANSACTION_TABLE_URI}`
        WHERE
        DATE(TX_TS) BETWEEN DATE_SUB("{DATAPROCESSING_END_DATE}", INTERVAL 15 DAY) AND "{DATAPROCESSING_END_DATE}"
        ) raw_tx
    LEFT JOIN 
        `{RAW_BQ_LABELS_TABLE_URI}` as raw_lb
    ON raw_tx.TX_ID = raw_lb.TX_ID),

    -- query to calculate CUSTOMER spending behaviour --------------------------------------------------------------------------------
    get_customer_spending_behaviour AS (
    SELECT
        TX_TS,
        TX_ID,
        CUSTOMER_ID,
        TERMINAL_ID,
        TX_AMOUNT,
        TX_FRAUD,
        
        # calc the number of customer tx over daily windows per customer (1, 7 and 15 days, expressed in seconds)
        COUNT(TX_FRAUD) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 86400 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_NB_TX_1DAY_WINDOW,
        COUNT(TX_FRAUD) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 604800 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_NB_TX_7DAY_WINDOW,
        COUNT(TX_FRAUD) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1209600 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_NB_TX_14DAY_WINDOW,
        
        # calc the customer average tx amount over daily windows per customer (1, 7 and 15 days, expressed in seconds, in dollars ($))
        AVG(TX_AMOUNT) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 86400 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW,
        AVG(TX_AMOUNT) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 604800 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW,
        AVG(TX_AMOUNT) OVER (PARTITION BY CUSTOMER_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1209600 PRECEDING
        AND CURRENT ROW ) AS CUSTOMER_ID_AVG_AMOUNT_14DAY_WINDOW,
    FROM get_raw_table)

    # Create the table with CUSTOMER and TERMINAL features ----------------------------------------------------------------------------
    SELECT
    PARSE_TIMESTAMP("%Y-%m-%d %H:%M:%S", FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", TX_TS, "UTC")) AS feature_ts,
    CUSTOMER_ID AS customer_id,
    CAST(CUSTOMER_ID_NB_TX_1DAY_WINDOW AS INT64) AS customer_id_nb_tx_1day_window,
    CAST(CUSTOMER_ID_NB_TX_7DAY_WINDOW AS INT64) AS customer_id_nb_tx_7day_window,
    CAST(CUSTOMER_ID_NB_TX_14DAY_WINDOW AS INT64) AS customer_id_nb_tx_14day_window,
    CAST(CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW AS FLOAT64) AS customer_id_avg_amount_1day_window,
    CAST(CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW AS FLOAT64) AS customer_id_avg_amount_7day_window,
    CAST(CUSTOMER_ID_AVG_AMOUNT_14DAY_WINDOW AS FLOAT64) AS customer_id_avg_amount_14day_window,
    FROM
    get_customer_spending_behaviour
    """
    run_bq_query(create_customer_batch_features_query)

    return "Done get_batch_data_bq"


def create_terminal_feature_table(
    DATAPROCESSING_END_DATE ,
    RAW_BQ_TRANSACTION_TABLE_URI,
    RAW_BQ_LABELS_TABLE_URI ,
    TERMINALS_BQ_TABLE_URI ):
    

    create_terminal_batch_features_query = f"""
    CREATE OR REPLACE TABLE `{TERMINALS_BQ_TABLE_URI}` AS
    WITH
    -- query to join labels with features -------------------------------------------------------------------------------------------
    get_raw_table AS (
    SELECT
        raw_tx.TX_TS,
        raw_tx.TX_ID,
        raw_tx.CUSTOMER_ID,
        raw_tx.TERMINAL_ID,
        raw_tx.TX_AMOUNT,
        raw_lb.TX_FRAUD
    FROM (
        SELECT
        *
        FROM
        `{RAW_BQ_TRANSACTION_TABLE_URI}`
        WHERE
        DATE(TX_TS) BETWEEN DATE_SUB("{DATAPROCESSING_END_DATE}", INTERVAL 15 DAY) AND "{DATAPROCESSING_END_DATE}"
        ) raw_tx
    LEFT JOIN 
        `{RAW_BQ_LABELS_TABLE_URI}` as raw_lb
    ON raw_tx.TX_ID = raw_lb.TX_ID),

    # query to calculate TERMINAL spending behaviour --------------------------------------------------------------------------------
    get_variables_delay_window AS (
    SELECT
        TX_TS,
        TX_ID,
        CUSTOMER_ID,
        TERMINAL_ID,
        
        # calc total amount of fraudulent tx and the total number of tx over the delay period per terminal (7 days - delay, expressed in seconds)
        SUM(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 604800 PRECEDING
        AND CURRENT ROW ) AS NB_FRAUD_DELAY,
        COUNT(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 604800 PRECEDING
        AND CURRENT ROW ) AS NB_TX_DELAY,
        
        # calc total amount of fraudulent tx and the total number of tx over the delayed window per terminal (window + 7 days - delay, expressed in seconds)
        SUM(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 691200 PRECEDING
        AND CURRENT ROW ) AS NB_FRAUD_1_DELAY_WINDOW,
        SUM(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1209600 PRECEDING
        AND CURRENT ROW ) AS NB_FRAUD_7_DELAY_WINDOW,
        SUM(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1814400 PRECEDING
        AND CURRENT ROW ) AS NB_FRAUD_14_DELAY_WINDOW,
        COUNT(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 691200 PRECEDING
        AND CURRENT ROW ) AS NB_TX_1_DELAY_WINDOW,
        COUNT(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1209600 PRECEDING
        AND CURRENT ROW ) AS NB_TX_7_DELAY_WINDOW,
        COUNT(TX_FRAUD) OVER (PARTITION BY TERMINAL_ID ORDER BY UNIX_SECONDS(TX_TS) ASC RANGE BETWEEN 1814400 PRECEDING
        AND CURRENT ROW ) AS NB_TX_14_DELAY_WINDOW,
    FROM get_raw_table),

    # query to calculate TERMINAL risk factors ---------------------------------------------------------------------------------------
    get_risk_factors AS (
    SELECT
        TX_TS,
        TX_ID,
        CUSTOMER_ID,
        TERMINAL_ID,
        # calculate numerator of risk index
        NB_FRAUD_1_DELAY_WINDOW - NB_FRAUD_DELAY AS TERMINAL_ID_NB_FRAUD_1DAY_WINDOW,
        NB_FRAUD_7_DELAY_WINDOW - NB_FRAUD_DELAY AS TERMINAL_ID_NB_FRAUD_7DAY_WINDOW,
        NB_FRAUD_14_DELAY_WINDOW - NB_FRAUD_DELAY AS TERMINAL_ID_NB_FRAUD_14DAY_WINDOW,
        # calculate denominator of risk index
        NB_TX_1_DELAY_WINDOW - NB_TX_DELAY AS TERMINAL_ID_NB_TX_1DAY_WINDOW,
        NB_TX_7_DELAY_WINDOW - NB_TX_DELAY AS TERMINAL_ID_NB_TX_7DAY_WINDOW,
        NB_TX_14_DELAY_WINDOW - NB_TX_DELAY AS TERMINAL_ID_NB_TX_14DAY_WINDOW,
        FROM
        get_variables_delay_window),

    # query to calculate the TERMINAL risk index -------------------------------------------------------------------------------------
    get_risk_index AS (
        SELECT
        TX_TS,
        TX_ID,
        CUSTOMER_ID,
        TERMINAL_ID,
        TERMINAL_ID_NB_TX_1DAY_WINDOW,
        TERMINAL_ID_NB_TX_7DAY_WINDOW,
        TERMINAL_ID_NB_TX_14DAY_WINDOW,
        # calculate the risk index
        (TERMINAL_ID_NB_FRAUD_1DAY_WINDOW/(TERMINAL_ID_NB_TX_1DAY_WINDOW+0.0001)) AS TERMINAL_ID_RISK_1DAY_WINDOW,
        (TERMINAL_ID_NB_FRAUD_7DAY_WINDOW/(TERMINAL_ID_NB_TX_7DAY_WINDOW+0.0001)) AS TERMINAL_ID_RISK_7DAY_WINDOW,
        (TERMINAL_ID_NB_FRAUD_14DAY_WINDOW/(TERMINAL_ID_NB_TX_14DAY_WINDOW+0.0001)) AS TERMINAL_ID_RISK_14DAY_WINDOW
        FROM get_risk_factors 
    )

    # Create the table with CUSTOMER and TERMINAL features ----------------------------------------------------------------------------
    SELECT
    PARSE_TIMESTAMP("%Y-%m-%d %H:%M:%S", FORMAT_TIMESTAMP("%Y-%m-%d %H:%M:%S", TX_TS, "UTC")) AS feature_ts,
    TERMINAL_ID AS terminal_id,
    CAST(TERMINAL_ID_NB_TX_1DAY_WINDOW AS INT64) AS terminal_id_nb_tx_1day_window,
    CAST(TERMINAL_ID_NB_TX_7DAY_WINDOW AS INT64) AS terminal_id_nb_tx_7day_window,
    CAST(TERMINAL_ID_NB_TX_14DAY_WINDOW AS INT64) AS terminal_id_nb_tx_14day_window,
    CAST(TERMINAL_ID_RISK_1DAY_WINDOW AS FLOAT64) AS terminal_id_risk_1day_window,
    CAST(TERMINAL_ID_RISK_7DAY_WINDOW AS FLOAT64) AS terminal_id_risk_7day_window,
    CAST(TERMINAL_ID_RISK_14DAY_WINDOW AS FLOAT64) AS terminal_id_risk_14day_window,
    FROM
    get_risk_index
    """
    run_bq_query(create_terminal_batch_features_query)

    return "Done get_batch_data_bq"

def create_real_time_features(    
    CUSTOMERS_BQ_TABLE_URI ,
    TERMINALS_BQ_TABLE_URI,
      ):
    

    initiate_real_time_customer_features_query = f"""
    ALTER TABLE `{CUSTOMERS_BQ_TABLE_URI}`
    ADD COLUMN customer_id_nb_tx_15min_window INT64,
    ADD COLUMN customer_id_nb_tx_30min_window INT64,
    ADD COLUMN customer_id_nb_tx_60min_window INT64,
    ADD COLUMN customer_id_avg_amount_15min_window FLOAT64,
    ADD COLUMN customer_id_avg_amount_30min_window FLOAT64,
    ADD COLUMN customer_id_avg_amount_60min_window FLOAT64;

    ALTER TABLE `{CUSTOMERS_BQ_TABLE_URI}`
    ALTER COLUMN customer_id_nb_tx_15min_window SET DEFAULT 0,
    ALTER COLUMN customer_id_nb_tx_30min_window SET DEFAULT 0,
    ALTER COLUMN customer_id_nb_tx_60min_window SET DEFAULT 0,
    ALTER COLUMN customer_id_avg_amount_15min_window SET DEFAULT 0,
    ALTER COLUMN customer_id_avg_amount_30min_window SET DEFAULT 0,
    ALTER COLUMN customer_id_avg_amount_60min_window SET DEFAULT 0;

    UPDATE `{CUSTOMERS_BQ_TABLE_URI}`
    SET customer_id_nb_tx_15min_window = 0,
        customer_id_nb_tx_30min_window  = 0,
        customer_id_nb_tx_60min_window  = 0, 
        customer_id_avg_amount_15min_window = 0,
        customer_id_avg_amount_30min_window  = 0,
        customer_id_avg_amount_60min_window  = 0
    WHERE TRUE; 
    """
    initiate_real_time_terminal_features_query = f"""
    ALTER TABLE `{TERMINALS_BQ_TABLE_URI}`
    ADD COLUMN terminal_id_nb_tx_15min_window INT64,
    ADD COLUMN terminal_id_nb_tx_30min_window INT64,
    ADD COLUMN terminal_id_nb_tx_60min_window INT64,
    ADD COLUMN terminal_id_avg_amount_15min_window FLOAT64,
    ADD COLUMN terminal_id_avg_amount_30min_window FLOAT64,
    ADD COLUMN terminal_id_avg_amount_60min_window FLOAT64;

    ALTER TABLE `{TERMINALS_BQ_TABLE_URI}`
    ALTER COLUMN terminal_id_nb_tx_15min_window SET DEFAULT 0,
    ALTER COLUMN terminal_id_nb_tx_30min_window SET DEFAULT 0,
    ALTER COLUMN terminal_id_nb_tx_60min_window SET DEFAULT 0,
    ALTER COLUMN terminal_id_avg_amount_15min_window SET DEFAULT 0,
    ALTER COLUMN terminal_id_avg_amount_30min_window SET DEFAULT 0,
    ALTER COLUMN terminal_id_avg_amount_60min_window SET DEFAULT 0;

    UPDATE `{TERMINALS_BQ_TABLE_URI}`
    SET terminal_id_nb_tx_15min_window = 0,
        terminal_id_nb_tx_30min_window  = 0,
        terminal_id_nb_tx_60min_window  = 0,
        terminal_id_avg_amount_15min_window = 0,
        terminal_id_avg_amount_30min_window = 0,
        terminal_id_avg_amount_60min_window  = 0
    WHERE TRUE; 
    """
    for query in [initiate_real_time_customer_features_query, initiate_real_time_terminal_features_query]:
        run_bq_query(query)

    return "Done get_batch_data_bq"



def create_featurestore(destination_project_id, REGION, BUCKET_NAME,
    FEATURESTORE_ID,
    ONLINE_STORAGE_NODES,
    CUSTOMER_ENTITY_ID, TERMINAL_ENTITY_ID,
    CUSTOMERS_BQ_TABLE_URI, FEATURE_TIME, TERMINALS_BQ_TABLE_URI
    ) ->  Featurestore : 
    vertex_ai.init(project=destination_project_id, location=REGION, staging_bucket=BUCKET_NAME)
    try:
        # Checks if there is already a Featurestore        
        ff_feature_store = vertex_ai.Featurestore(f"{FEATURESTORE_ID}")
        print(f"""The feature store {FEATURESTORE_ID} already exists.""")
    except:
        # Creates a Featurestore
        print(f"""Creating new feature store {FEATURESTORE_ID}.""")
        ff_feature_store = Featurestore.create(
            featurestore_id=f"{FEATURESTORE_ID}",
            online_store_fixed_node_count=ONLINE_STORAGE_NODES,
            labels={"team": "cymbal_bank", "app": "fraudfinder"},
            sync=True,
        )
    try:
        # get entity type, if it already exists
        customer_entity_type = ff_feature_store.get_entity_type(entity_type_id=CUSTOMER_ENTITY_ID)
    except:
        # else, create entity type
        customer_entity_type = ff_feature_store.create_entity_type(
            entity_type_id=CUSTOMER_ENTITY_ID, description="Customer Entity", sync=True
        )

    customer_feature_configs = {
        "customer_id_nb_tx_1day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last day",
            "labels": {"status": "passed"},
        },
        "customer_id_nb_tx_7day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last 7 days",
            "labels": {"status": "passed"},
        },
        "customer_id_nb_tx_14day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last 14 days",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_1day_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last day",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_7day_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 7 days",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_14day_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 14 days",
            "labels": {"status": "passed"},
        },
        "customer_id_nb_tx_15min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last 15 minutes",
            "labels": {"status": "passed"},
        },
        "customer_id_nb_tx_30min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last 30 minutes",
            "labels": {"status": "passed"},
        },
        "customer_id_nb_tx_60min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the customer in the last 60 minutes",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_15min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 15 minutes",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_30min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 30 minutes",
            "labels": {"status": "passed"},
        },
        "customer_id_avg_amount_60min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 60 minutes",
            "labels": {"status": "passed"},
        },
    }
    customer_feature_ids = customer_entity_type.batch_create_features(
        feature_configs=customer_feature_configs, sync=True )
   
    try:
        # get entity type, if it already exists
        terminal_entity_type = ff_feature_store.get_entity_type(entity_type_id=TERMINAL_ENTITY_ID)
    except:
        # else, create entity type
        terminal_entity_type = ff_feature_store.create_entity_type(
        entity_type_id=TERMINAL_ENTITY_ID, description="Terminal Entity", sync=True
    )
    
    terminal_feature_configs = {
        "terminal_id_nb_tx_1day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the last day",
            "labels": {"status": "passed"},
        },
        "terminal_id_nb_tx_7day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the 7 days",
            "labels": {"status": "passed"},
        },
        "terminal_id_nb_tx_14day_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the 14 days",
            "labels": {"status": "passed"},
        },
        "terminal_id_risk_1day_window": {
            "value_type": "DOUBLE",
            "description": "Risk score calculated average number of frauds on the terminal in the last day",
            "labels": {"status": "passed"},
        },
        "terminal_id_risk_7day_window": {
            "value_type": "DOUBLE",
            "description": "Risk score calculated average number of frauds on the terminal in the last 7 days",
            "labels": {"status": "passed"},
        },
        "terminal_id_risk_14day_window": {
            "value_type": "DOUBLE",
            "description": "Risk score calculated average number of frauds on the terminal in the last 14 day",
            "labels": {"status": "passed"},
        },
        "terminal_id_nb_tx_15min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the last 15 minutes",
            "labels": {"status": "passed"},
        },
        "terminal_id_nb_tx_30min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the last 30 minutes",
            "labels": {"status": "passed"},
        },
        "terminal_id_nb_tx_60min_window": {
            "value_type": "INT64",
            "description": "Number of transactions by the terminal in the last 60 minutes",
            "labels": {"status": "passed"},
        },
        "terminal_id_avg_amount_15min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 15 minutes",
            "labels": {"status": "passed"},
        },
        "terminal_id_avg_amount_30min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 30 minutes",
            "labels": {"status": "passed"},
        },
        "terminal_id_avg_amount_60min_window": {
            "value_type": "DOUBLE",
            "description": "Average spending amount in the last 60 minutes",
            "labels": {"status": "passed"},
        },
    }

    terminal_feature_ids = terminal_entity_type.batch_create_features(
        feature_configs=terminal_feature_configs, sync=True
    )

    CUSTOMERS_FEATURES_IDS = [feature.name for feature in customer_feature_ids.list_features()]
    CUSTOMER_BQ_SOURCE_URI = f"bq://{CUSTOMERS_BQ_TABLE_URI}"
    CUSTOMER_ENTITY_ID_FIELD = "customer_id"
    customer_entity_type.ingest_from_bq(
        feature_ids=CUSTOMERS_FEATURES_IDS,
        feature_time=FEATURE_TIME,
        bq_source_uri=CUSTOMER_BQ_SOURCE_URI,
        entity_id_field=CUSTOMER_ENTITY_ID_FIELD,
        disable_online_serving=False,
        worker_count=10,
        sync=True,
    )

    
    TERMINALS_FEATURES_IDS = [feature.name for feature in terminal_feature_ids.list_features()]
    TERMINALS_BQ_SOURCE_URI = f"bq://{TERMINALS_BQ_TABLE_URI}"
    TERMINALS_ENTITY_ID_FIELD = "terminal_id"   
    terminal_entity_type.ingest_from_bq(
        feature_ids=TERMINALS_FEATURES_IDS,
        feature_time=FEATURE_TIME,
        bq_source_uri=TERMINALS_BQ_SOURCE_URI,
        entity_id_field=TERMINALS_ENTITY_ID_FIELD,
        disable_online_serving=False,
        worker_count=10,
        sync=True,
    )
