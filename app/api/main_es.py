from fastapi import FastAPI, Depends, HTTPException
from fastapi import File, UploadFile
from elasticsearch_dsl import connections

from app.api.config import ES_URL
from app.api.upload_weights_service_es import upload_weights, MinioParam
from app.api.upload_es_kb_service import upload_es_kb_service, EsParam
from app.api.feedback_service_es import upload_feedback, FeedbackRequest
from app.api.qa_service_es import make_query, query_request, load_index

from src.models import GoldenRetriever
from src.encoders import USEEncoder

app = FastAPI(title='HotDocs Golden Retriever API', description='Answer retrieval engine. Feed it your documents and get back answers')

# connect to Elasticsearch 
connections.create_connection(hosts=[ES_URL])

# initialize model and simple neighbors index
enc = USEEncoder()
gr = GoldenRetriever(enc)  # use gr_2.restore_encoder(save_dir=save_dir) if instantiating a saved model 
index = load_index()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/query")
def query(request: query_request):
    """
    Main function for User to make requests to. 

    :type query: str
    :type top_k: int, default 5
    :param query: query string contains their natural question
    :param top_k: Number of top responses to query. Currently kept at 5
    :return: a list that contains top_k string responses
    :return: an integer query_id that contains id of the request to be used for when they give feedback
    """
    try:
        resp, query_id = make_query(request, gr, index)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return {'resp': resp, 'query_id': query_id}


@app.post("/upload_weights")
async def save_weights(minio_param: MinioParam = Depends(), file: UploadFile = File(...)):
    """
    Upload finetuned weights to an minio s3 storage container. 
    Include minio params as form data along with file for uploading 

    Sample form data:

    .. highlight:: python
    .. code-block:: python
       
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
    Index QnA datasets into Elasticsearch for downstream finetuning and serving

    :type es_param: str
    :param es_param: can be 'query' or 'response'. Use to compare statements
    :param csv_file: csv file containing queries and responses. 
                  CSV files should have the following columns: 
                  [ans_id, ans_str, context_str (optional), query_str, query_id]
    :return: response dictionary as such: {'message': 'success', 'number of docs': counter}
    :return: query_id, an int that contains id of the request to be used for when they give feedback
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

    :param feedback_request: See declared basemodel below. 
    :return: A dictionary indicated successful transaction as such {'resp': 'updated'}

    .. highlight:: python
    .. code-block:: python

        class FeedbackRequest(BaseModel):
        query_id: str
        is_correct: List[bool]
    """
    try:
        message = upload_feedback(feedback_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    return message
