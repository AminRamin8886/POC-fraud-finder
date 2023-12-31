
# Vertex Pipelines
from typing import NamedTuple

from kfp.v2.dsl import (   
    component
)


from google.cloud import aiplatform as vertex_ai
from google_cloud_pipeline_components import aiplatform as vertex_ai_components
from kfp.v2.google.client import AIPlatformClient as VertexAIClient


 
@component(
    base_image="python:3.7",
    packages_to_install=["google-cloud-aiplatform==1.21.0"],
)
def ingest_features_gcs(
    project_id: str,
    region: str,
    bucket_name: str,
    feature_store_id: str,
    read_instances_uri: str,
) -> NamedTuple("Outputs", [("snapshot_uri_paths", str),],):
    # Libraries --------------------------------------------------------------------------------------------------------------------------
    from datetime import datetime
    import glob
    import urllib
    import json

    # Feature Store
    from google.cloud.aiplatform import Featurestore, EntityType, Feature

    # Variables --------------------------------------------------------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    api_endpoint = region + "-aiplatform.googleapis.com"
    bucket = urllib.parse.urlsplit(bucket_name).netloc
    export_uri = (
        f"{bucket_name}/data/snapshots/{timestamp}"  # format as new gsfuse requires
    )
    export_uri_path = f"/gcs/{bucket}/data/snapshots/{timestamp}"
    customer_entity = "customer"
    terminal_entity = "terminal"
    serving_feature_ids = {customer_entity: ["*"], terminal_entity: ["*"]}

    print(timestamp)
    print(bucket)
    print(export_uri)
    print(export_uri_path)
    print(customer_entity)
    print(terminal_entity)
    print(serving_feature_ids)

    # Main -------------------------------------------------------------------------------------------------------------------------------

    ## Define the feature store resource path
    feature_store_resource_path = (
        f"projects/{project_id}/locations/{region}/featurestores/{feature_store_id}"
    )
    print("Feature Store: \t", feature_store_resource_path)

    ## Run batch job request
    try:
        ff_feature_store = Featurestore(feature_store_resource_path)
        ff_feature_store.batch_serve_to_gcs(
            gcs_destination_output_uri_prefix=export_uri,
            gcs_destination_type="csv",
            serving_feature_ids=serving_feature_ids,
            read_instances_uri=read_instances_uri,
            pass_through_fields=["tx_fraud", "tx_amount"],
        )
    except Exception as error:
        print(error)

    # Store metadata
    snapshot_pattern = f"{export_uri_path}/*.csv"
    snapshot_files = glob.glob(snapshot_pattern)
    snapshot_files_fmt = [p.replace("/gcs/", "gs://") for p in snapshot_files]
    snapshot_files_string = json.dumps(snapshot_files_fmt)

    component_outputs = NamedTuple(
        "Outputs",
        [
            ("snapshot_uri_paths", str),
        ],
    )

    print(snapshot_pattern)
    print(snapshot_files)
    print(snapshot_files_fmt)
    print(snapshot_files_string)

    return component_outputs(snapshot_files_string)