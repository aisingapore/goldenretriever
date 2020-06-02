from fastapi import FastAPI, Depends, HTTPException
from fastapi import File, UploadFile
from elasticsearch_dsl import connections

from app.api.config import ES_URL
from app.api.upload_weights_service_es import upload_weights, MinioParam
from app.api.upload_es_kb_service import upload_es_kb_service, EsParam
from app.api.feedback_service_es import upload_feedback, FeedbackRequest
from app.api.qa_service_es import make_query

from src.models import GoldenRetriever
from src.encoders import USEEncoder

app = FastAPI(title='HotDocs Golden Retriever API', description='Answer retrieval engine. Feed it your documents and get back answers')

# connect to Elasticsearch 
connections.create_connection(hosts=[ES_URL])

# initialize model 
enc = USEEncoder()
gr = GoldenRetriever(enc)  # use gr_2.restore_encoder(save_dir=save_dir) if instantiating a saved model 

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query/{query_string}/{k}")
def query(query_string: str, k: int = 5):
    """
    Main function for User to make requests to. 

    Args:
    -----
        query: (str) query string contains their natural question
        top_k: (int, default 5) Number of top responses to query. Currently kept at 5

    Return:
    -------
        reply: (list) contains top_k string responses
        query_id: (int) contains id of the request to be used for when they give feedback
    """
    try:
        resp, query_id = make_query(query_string, gr, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return {'resp': resp, 'query_id': query_id}


@app.post("/upload_weights")
async def save_weights(minio_param: MinioParam = Depends(), file: UploadFile = File(...)):
    """
    Upload finetuned weights to an minio s3 storage container
    Include minio params as form data along with file for uploading 
    Sample form data:
            {
             'minio_url': MINIO_URL
             'minio_access_key': MINIO_ACCESS_KEY
             'minio_secret_key': MINIO_SECRET_KEY
             'bucket_name': BUCKET_NAME,
             'object_name': OBJECT_NAME
            } 
    """
    try:
        message = upload_weights(minio_param, file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return {'message': message}


@app.post("/upload_es_kb")
async def upload_kb(es_param: EsParam = Depends(), csv_file: UploadFile = File(...)):
    """
    index QnA datasets into Elasticsearch for downstream finetuning and serving

    Args:
    -----
        es_param: (str) ES_URL
        csv_file: csv file containing queries and responses. 
                  CSV files should have the following columns: 
                  [ans_id, ans_str, context_str (optional), query_str, query_id]

    Return:
    -------
        resp: (dict) {'message': 'success', 'number of docs': counter}
        query_id: (int) contains id of the request to be used for when they give feedback
    """
    try: 
        message = upload_es_kb_service(es_param, csv_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return message


@app.post("/feedback")
def save_feedback(feedback_request: FeedbackRequest):
    """
    Update ES querylog index with user feedback

    Args:
    -----
        feedback_request: (pydantic BaseModel) 
        class FeedbackRequest(BaseModel):
            query_id: str
            is_correct: List[bool]

    Return:
    -------
        resp: (dict) {'resp': 'updated'}
    """
    try:
        message = upload_feedback(feedback_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return message
