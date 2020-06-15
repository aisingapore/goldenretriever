from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists

class MinioClient():
    def __init__(self, url_endpoint, access_key, secret_key):
        self.client = Minio(url_endpoint, access_key, secret_key, secure=False)

    def make_bucket(self, bucket_name):
        try:
            self.client.make_bucket(bucket_name)
        except BucketAlreadyExists as err:
            print(err)
        except BucketAlreadyOwnedByYou as err:
            print(err)
        except ResponseError as err:
            print(err)

    def rm_bucket(self, bucket_name):
        try:
            self.client.remove_bucket(bucket_name)
        except ResponseError as err:
            print(err)

    def upload_model_weights(self, bucket_name, model_obj_name, model_file_path):
        try:
            print(self.client.fput_object(bucket_name, model_obj_name, model_file_path))
        except ResponseError as err:
            print(err)

    def download_model_weights(self, bucket_name, model_obj_name, model_file_path):
        try:
            print(self.client.fget_object(bucket_name, model_obj_name, model_file_path))
        except ResponseError as err:
            print(err)

    def upload_emb_index(self, bucket_name, emb_obj_name, emb_file_path):
        try:
            print(self.client.fput_object(bucket_name, emb_obj_name, emb_file_path))
        except ResponseError as err:
            print(err)
        
    def download_emb_index(self, bucket_name, emb_obj_name, emb_file_path):
        try:
            print(self.client.fget_object(bucket_name, emb_obj_name, emb_file_path))
        except ResponseError as err:
            print(err)

