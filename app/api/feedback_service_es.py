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
    is_correct: List[int]


def upload_feedback(feedback_request):

    def send_feedback_es(feedback_request, feedback_timestamp):
        # parse is_correct 
        is_correct = feedback_request.is_correct
        is_correct = is_correct + [False] * (5 - len(is_correct)) if len(is_correct) < 5 else is_correct  # ensures 5 entries

        # get ES querylog entry based on query id 
        query_id = feedback_request.query_id
        s = QueryLog.search().query('match', query_id=query_id)
        res = s.execute()
        log_id = res.meta.id

        # update ES entry
        entry = QueryLog.get(id=log_id)
        entry.update(is_correct=is_correct, feedback_timestamp=feedback_timestamp)
        update_res = entry.execute()
        return update_res

    feedback_timestamp = datetime.now()
    update_res = send_feedback_es(feedback_request, feedback_timestamp)
    return update_res