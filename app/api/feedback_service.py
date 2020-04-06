import requests
import datetime
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds

from db_handler import get_last_insert_ids, extract_qa_pair_based_on_idx, get_kb_id_ref, get_permissions, ensure_connection
from exceptions import InvalidUsage

def save_feedback(request, conn, cursor):

    def parse_req(request):
        request_timestamp = datetime.datetime.now()
        request_dict = request.get_json()

        if not all([key in request_dict.keys() for key in ['query_id', 'is_correct']]):
            raise InvalidUsage("request requires 'query_id', 'is_correct")
        if not (type(request_dict["is_correct"]) == list) & all([type(feedback_)==int for feedback_ in request_dict['is_correct']]):
            raise InvalidUsage("'is_correct' is to contain a list of integers, e.g. {'query_id':45, 'is_correct':[0,1,0,0,0]} indicates that the second ranked answer was correct")

        return request_dict, request_timestamp

    def send_feedback_to_sql(request_dict, request_timestamp, conn, cursor):
        # 1. parse the request
        query_id = request_dict["query_id"]
        is_correct = request_dict["is_correct"]
        is_correct = is_correct+[False]*(5-len(is_correct)) if len(is_correct) < 5 else is_correct # ensures 5 entries

        # log the request in SQL
        rowinfo = [request_timestamp]
        rowinfo.append(query_id)
        rowinfo.extend(is_correct[:5]) # ensures only 5 values are logged

        conn, cursor = ensure_connection(conn, cursor)
        cursor.execute('INSERT INTO feedback_log VALUES (?, ?, ?, ?, ?, ?, ?)', rowinfo)
        cursor.commit()

    request_dict, request_timestamp = parse_req(request)
    send_feedback_to_sql(request_dict, request_timestamp, conn, cursor)