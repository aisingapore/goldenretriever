from minio import Minio
from minio.error import ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists
from dotenv import load_dotenv

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

    def upload_model_weights(self, bucket_name, model_obj_name, model_file_path):
        try:
            print(self.client.fput_object(bucket_name, model_obj_name, model_file_path))
        except ResponseError as err:
            print(err)

    def download_model_weights(self, bucket_name, model_obj_name):
        try:
            print(self.client.get_object(bucket_name, model_obj_name))
        except ResponseError as err:
            print(err)

    def upload_emb_index(self, bucket_name, emb_obj_name, emb_file_path):
        try:
            print(self.client.fput_object(bucket_name, emb_obj_name, emb_file_path))
        except ResponseError as err:
            print(err)
        
    def download_emb_index(self, bucket_name, emb_obj_name):
        try:
            print(self.client.get_object(bucket_name, emb_obj_name))
        except ResponseError as err:
            print(err)


if __name__=='__main__':
    from dotenv import load_dotenv
    load_dotenv()
    file_path = '/Users/nus/Downloads/use_model/1.tar.gz'
    model_obj_name = 'use_model'
    bucket_name = 'pdpa'
    minio_client = MinioClient("URL_ENDPOINT", "ACCESS_KEY", "SECRET_KEY")
    minio_client.upload_model_weights(bucket_name, model_obj_name, file_path)
