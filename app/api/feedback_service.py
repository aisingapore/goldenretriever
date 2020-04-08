import requests
import datetime
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds
from pydantic import BaseModel
from typing import List

from db_handler import get_last_insert_ids, extract_qa_pair_based_on_idx, get_kb_id_ref, get_permissions, ensure_connection
from exceptions import InvalidUsage

class feedback_request(BaseModel):
    query_id: str
    is_correct: List[int]

def save_feedback(request, conn, cursor):

    def send_feedback_to_sql(request, request_timestamp, conn, cursor):
        # 1. parse the request
        query_id = request.query_id
        is_correct = request.is_correct
        is_correct = is_correct+[False]*(5-len(is_correct)) if len(is_correct) < 5 else is_correct # ensures 5 entries

        # log the request in SQL
        rowinfo = [request_timestamp]
        rowinfo.append(query_id)
        rowinfo.extend(is_correct[:5]) # ensures only 5 values are logged

        conn, cursor = ensure_connection(conn, cursor)
        cursor.execute('INSERT INTO feedback_log VALUES (?, ?, ?, ?, ?, ?, ?)', rowinfo)
        cursor.commit()

    request_timestamp = datetime.datetime.now()
    send_feedback_to_sql(request, request_timestamp, conn, cursor)