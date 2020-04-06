import requests
import datetime
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds

from db_handler import get_last_insert_ids, extract_qa_pair_based_on_idx, get_kb_id_ref, get_permissions, ensure_connection
from exceptions import InvalidUsage

def make_query(request, gr, conn, cursor, permissions, get_kb_dir_id, get_kb_raw_id):
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

    def parse_req():
        # 1. parse the request and get timestamp
        request_timestamp = datetime.datetime.now()
        request_dict = request.get_json()
        
        if not all([key in request_dict.keys() for key in ['query', 'kb_name']]):
            raise InvalidUsage()

        return request_dict, request_timestamp
    
    def get_inference(query_string, kb_name):
        # 2. model inference
        reply, reply_index = gr.make_query(query_string, 
                                        # top_k=int(top_k), 
                                        top_k = 5,
                                        index=True, kb_name=kb_name)
        return reply, reply_index

    def log_req(request_dict, request_timestamp, reply_index, conn, cursor):
        # 3. log the request in SQL
        # query log has the following columns
        # id, created_at, query_string, user_id, kb_dir_id, kb_raw_id, Answer1, Answer2, Answer3, Answer4, Answer5
        HASHKEY = request_dict.get('hashkey')
        query_string = request_dict["query"]
        kb_name = request_dict["kb_name"]
        
        rowinfo = [request_timestamp, query_string] 
        # append user_id
        logged_user_id = permissions.loc[HASHKEY].user_id.iloc[-1] if HASHKEY in permissions.index else None
        rowinfo.append(logged_user_id) 
        # append kb_dir_id
        rowinfo.append(get_kb_dir_id[kb_name])   
        # append kb_raw_id
        rowinfo.append(get_kb_raw_id[kb_name])
        # returned answers clause_id
        rowinfo.extend(gr.kb[kb_name].responses.clause_id.iloc[reply_index].tolist())

        conn, cursor = ensure_connection(conn, cursor)
        cursor.execute('INSERT INTO query_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rowinfo)
        cursor.commit()

        return conn, cursor

    def get_request_id(cursor):
        # 4. Return response to user
        # return id of latest log request to user for when they give feedback
        current_request_id = get_last_insert_ids(cursor)
        return current_request_id

    request_dict, request_timestamp = parse_req()
    reply, reply_index = get_inference(request_dict['query'], request_dict['kb_name'])
    conn, cursor = log_req(request_dict, request_timestamp, reply_index, conn, cursor)
    current_request_id = get_request_id(cursor)

    return reply, current_request_id