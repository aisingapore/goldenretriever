#!usr/bin/bash 

# init dvc 
dvc init 

# add minio container running locally as a remote 
dvc remote add -d myremote s3://mybucket/dvc-storage 
dvc remote modify myremote endpointurl http://localhost:9001
export AWS_ACCESS_KEY_ID=changeme
export AWS_SECRET_ACCESS_KEY=changeme

#upload cache to minio. make sure bucket has already been created 
dvc push 