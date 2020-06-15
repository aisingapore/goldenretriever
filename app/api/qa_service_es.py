import os 
from os.path import join
from os import getenv
from dotenv import load_dotenv
import uuid
from pydantic import BaseModel

from src.minio_handler import MinioClient
from src.prebuilt_index import SimpleNNIndex

from app.api.config import QueryLog, DOTENV_PATH, INDEX_BUCKET, INDEX_PICKLE, INDEX_FILE, INDEX_PREFIX, INDEX_FOLDER, INDEX_PICKLE_PATH, INDEX_FILE_PATH

load_dotenv(DOTENV_PATH)

class query_request(BaseModel):
    query: str
    k: int 

def load_index():
    """
    Load prebuilt vector index for nearest neighbors lookup
    :returns: Simple Nearest Neighbors index
    """
    if not os.path.exists(INDEX_FOLDER):
        os.makedirs(INDEX_FOLDER)
        minio_client = MinioClient(getenv('MINIO_URL'), getenv('ACCESS_KEY'), getenv('MINIO_SECRET_KEY'))
        minio_client.download_emb_index(INDEX_BUCKET, INDEX_PICKLE, INDEX_PICKLE_PATH)
        minio_client.download_emb_index(INDEX_BUCKET, INDEX_FILE, INDEX_FILE_PATH)
    index = SimpleNNIndex.load(join(INDEX_FOLDER, INDEX_PREFIX))
    return index


def get_inference(query_string, gr_obj, index, k=5):
    """
    Return respose to user's question
    :param query_string: question as str
    :param query_encoder: encoder for returning query embeddings
    :param k: number of responses to return
    :param index: name of prebuilt nearest neighbors index  
    :returns: top K responses that have highest similarity with question
    """
    query_embeddings = gr_obj.encoder.encode(query_string, string_type='query')
    resps = index.query(query_embeddings[0], k)
    return resps 


def log_request(query_string, resps):
    """
    Log user queries in Elasticsearch
    :param query: users input query
    :param responses: list of responses returned
    :param feedback: list of true / false responses returned by user   
    :return: True if query was logged successfully
    """
    id = uuid.uuid1()
    ql = QueryLog(text=query_string, resps=resps, query_id=id)
    ql.save()
    print('query saved')
    return id


def make_query(query_request, query_encoder, index):
    """
    Return respose to user's question
    :param query_string: question as str
    :param query_encoder: encoder for returning query embeddings
    :param k: number of responses to return
    :returns: top K responses that have highest similarity with question
    """
    # index = load_index()
    resp = get_inference(query_request.query, query_encoder, index, query_request.k)
    query_id = log_request(query_request.query, resp)
    return resp, query_id
