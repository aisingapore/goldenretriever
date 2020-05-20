"""
Version:
--------
0.1 16th May 2020

Usage:
------
Update query log index is_correct field with user feedback

Feedback is a list of booleans len(5) indicating whether responses are True or False
"""
from pydantic import BaseModel
from typing import List
from datetime import datetime
from app.api.config import QueryLog


class FeedbackRequest(BaseModel):
    query_id: str
    is_correct: List[bool]


def upload_feedback(feedback_request):

    def send_feedback_es(feedback_request):
        # parse is_correct 
        is_correct = feedback_request.is_correct
        print(is_correct)
        is_correct = is_correct + [False] * (5 - len(is_correct)) if len(is_correct) < 5 else is_correct  # ensures 5 entries

        # get ES querylog entry based on query id 
        query_id = feedback_request.query_id
        s = QueryLog.search().query('match', query_id=query_id)
        res = s.execute()
        log_id = res[0].meta.id

        # update ES entry
        entry = QueryLog.get(id=log_id)
        resp = entry.update(is_correct=is_correct, feedback_timestamp=datetime.now())
        return {'resp': resp}

    update_res = send_feedback_es(feedback_request)
    return update_res