from src.minio_handler import MinioClient
from fastapi import Form


class MinioParam:
    def __init__(self, minio_url: str = Form(...), minio_access_key: str = Form(...), 
                 minio_secret_key: str = Form(...), bucket_name: str = Form(...), object_name: str = Form(...)):
        self.minio_url = minio_url
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.bucket_name = bucket_name
        self.object_name = object_name


def upload_weights(params, files=None):
    """
    Upload finetuned weights to an minio object storage container
    args:
    ----
        minio_url: (str) url endpoint for minio
        minio_access_key: (str) minio access key for authorized access to object storage
        minio_secret_key: (str) minio secret key for authorized access to object storage
        bucket_name: (str) name of newly created bucket that will store weights
        object_name: (str) name to be used for the newly stored weights in the container
    Sample json body:
        {
         'minio_url': MINIO_URL
         'minio_access_key': MINIO_ACCESS_KEY
         'minio_secret_key': MINIO_SECRET_KEY
         'bucket_name': BUCKET_NAME,
         'object_name': OBJECT_NAME
        } 
    """
    MINIO_URL = params.minio_url
    MINIO_ACCESS_KEY = params.minio_access_key
    MINIO_SECRET_KEY = params.minio_secret_key
    BUCKET_NAME = params.bucket_name
    OBJECT_NAME = params.object_name
    # create minio client that will call minio instance 
    minio_client = MinioClient(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)
    # create a container. error will be raised if bucket already exists
    existing_buckets = [b.name for b in minio_client.client.list_buckets()]
    if BUCKET_NAME not in existing_buckets:
        minio_client.make_bucket(BUCKET_NAME)
    
    # upload file
    if files is not None:
        # print(files.filename)
        minio_client.upload_model_weights(BUCKET_NAME, OBJECT_NAME, files.filename)
    return 'success'