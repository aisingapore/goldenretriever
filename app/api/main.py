"""
GoldenRetriever API

Run from root folder:
python app/api/main.py
"""
import os
import datetime
import argparse
import sys
import sqlite3
sys.path.append('')
import requests

import uvicorn
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds
from fastapi import FastAPI, Request, Depends
from starlette.datastructures import FormData

from src.models import GoldenRetriever
from src.data_handler.kb_handler import kb_handler
from app.api.db_handler import get_last_insert_ids, extract_qa_pair_based_on_idx, get_kb_id_ref, get_permissions, ensure_connection
from app.api.exceptions import InvalidUsage
from app.api.qa_service import make_query, query_request
from app.api.feedback_service import save_feedback, feedback_request
from app.api.upload_kb_service import upload_knowledge_base_to_sql, upload_kb_request
from app.api.upload_weights_service import upload_weights

app = FastAPI()

try:
    # In production mode
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", "--credentials", dest='dir',
                        default='./db_cnxn_str.txt',
                        help="directory of the pyodbc password string")
    args = parser.parse_args()

    conn_path = args.dir
    conn_str = open(conn_path, 'r').read()
except SystemExit:
    # In test mode
    # conn_path will be supplied within the apis
    conn_path = ""

    # during tests, kbs in production database will be loaded
    # hence production database conn_str will be used
    # conn_str to be declared via gitlab ci/cd settings
    conn_str = os.environ["CONN_STR"]

# nrf kb needs to be loaded for test purposes
kbh = kb_handler()
kbs = kbh.load_sql_kb(cnxn_str=conn_str, kb_names=['PDPA', 'nrf'])

gr = GoldenRetriever()
# gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
gr.load_kb(kbs)


def get_common_params():

    conn = pyodbc.connect(conn_str)

    get_kb_dir_id, get_kb_raw_id = get_kb_id_ref(conn)
    permissions = get_permissions(conn)

    return {"conn": conn, "conn_path": conn_path, "get_kb_dir_id": get_kb_dir_id, "get_kb_raw_id": get_kb_raw_id, "permissions": permissions}


@app.post("/query")
async def make_query_endpoint(request: query_request, commons: dict = Depends(get_common_params)):
    """
    Main function for User to make requests to. 

    Args:
    -----
        hashkey: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        query: (str) query string contains their natural question
        kb_name: (str) Name of knowledge base to query
        top_k: (int, default 5) Number of top responses to query. Currently kept at 5

    Return:
    -------
        reply: (list) contains top_k string responses
        query_id: (int) contains id of the request to be used for when they give feedback
    """

    conn = commons["conn"]
    cursor = conn.cursor()

    get_kb_dir_id = commons["get_kb_dir_id"]
    get_kb_raw_id = commons["get_kb_raw_id"]
    permissions = commons["permissions"]

    reply, current_request_id = make_query(request, gr, conn, cursor, permissions, get_kb_dir_id, get_kb_raw_id)

    return {"responses": reply, "query_id": current_request_id}


@app.post("/feedback")
async def save_feedback_endpoint(request: feedback_request, commons: dict = Depends(get_common_params)):
    """
    Retrieve feedback from end users

    args:
    ----
        query_id: (int) specifies the query to raise feedback for
        is_correct: (list) list fo booleans for true or false
    """

    conn = commons["conn"]
    cursor = conn.cursor()

    save_feedback(request, conn, cursor)

    return {"message":"Success"}


@app.post("/knowledge_base")
async def upload_knowledge_base_to_sql_endpoint(request : upload_kb_request, commons: dict = Depends(get_common_params)):
    """
    Receive knowledge bases from users
    
    args:
    ----
        hashkey: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        kb_name: (str) Name of knowledge base to save as
        kb: (dict) contains the responses, queries and mappings
                   where mapping maps the indices of (question, answer)


    Sample json body & sample kb:
        {
         'hashkey': HASHKEY,
         'kb_name':'test1',
         'kb':{'responses': ["I'm 21 years old", 
                             "I hate mondays"],
               'contexts': ["Bob", "Gary"],
               'queries': ["What do you not love?", 
                           "How old are you?"],
               'mapping': [(0,1), (1,0)]
              }
        } 
    """

    # conn_path will be changed to sqlite db if in test mode
    conn_path = commons["conn_path"]
    conn = commons["conn"]
    cursor = conn.cursor()

    get_kb_dir_id = commons["get_kb_dir_id"]
    get_kb_raw_id = commons["get_kb_raw_id"]
    permissions = commons["permissions"]

    kb_name = upload_knowledge_base_to_sql(request, conn, cursor, get_kb_dir_id, get_kb_raw_id, permissions)

    # load knowledge base into cached model
    kbs = kbh.load_sql_kb(cnxn_path=conn_path, kb_names=[kb_name])

    gr.load_kb(kbs)

    return {"message":"Success"}


@app.post("/upload_weights")
async def upload_weights_endpoint(request : Request):
    """
    Upload finetuned weights to an azure blob storage container

    To handle both JSON and files in the request in fastAPI, 
    we may use Request.form():
    https://github.com/tiangolo/fastapi/issues/143
    
    args:
    ----
        conn_str: (str) connection string for authorized access to blob storage
        container_name: (str) name of newly created container that will store weights
        blob_name: (str) name to be used for the newly stored weights in the container


    Sample json body:
        {
         'conn_str': CONN_STR,
         'container_name': CONTAINER_NAME,
         'blob_name': BLOB_NAME
        } 
    """
    request_form = await request.form()
    files = await request_form['file'].read()

    message = upload_weights(request_form, files)
    
    return {"message":message}

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=5000)