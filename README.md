1.	Create source code repository includes the following items
    - Python pipelines workflow source code where the pipeline workflow is defined
    - The Python pipeline components source code and the corresponding component specification files for the different pipeline components such as data extraction, data validation, data transformation, model training, model validation, package model/data, and model deployment.
    - Dockerfiles that are required to create Docker container images, one for each pipeline component.
2.	Create Cloud Build steps
    - The source code repository is copied to the Cloud Build runtime environment, under the /workspace directory.
    - Build Docker container images ( one for each pipeline component). The images are tagged with the $COMMIT_SHA parameter.
    - The Docker container images are uploaded to the Artifact Registry.
    - The image URL is updated in each of the component.yaml files with the created and tagged Docker container images.
    - The pipeline workflow is compiled to produce the pipeline.json file.
    - The pipeline.json file is uploaded to GCS bucket.
3.	Create Cloud Function to fetch pipeline definition from GCS bucket and Run pipeline
 
 
![Arch](/images/POCARCH.PNG)
