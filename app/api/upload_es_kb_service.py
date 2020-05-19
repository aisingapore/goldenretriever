"""
Version:
--------
0.1 11th May 2020

Usage:
------
Script to handle indexing of QnA datasets into Elasticsearch for downstream finetuning and serving
- Connect and upload NRF Documents to Elasticsearch

CSV files should have the following columns: 
[ans_id, ans_str, context_str (optional), query_str, query_id]

To see how index is built, refer to src.elasticsearch.create_doc_index.py
"""

from elasticsearch_dsl import Document, InnerDoc, Date, Nested, Keyword, Text, Integer
import pandas as pd 
from datetime import datetime
from pydantic import BaseModel


class EsParam(BaseModel):
    index_name: str


def upload_es_kb_service(es_param, csv_file):

    class QA(InnerDoc):
        ans_id = Integer()
        ans_str = Text(fields={'raw': Keyword()})
        query_id = Integer()
        query_str = Text()

    class Doc(Document):
        doc = Text()
        created_at = Date()
        qa_pair = Nested(QA)

        class Index:
            name = es_param.index_name

        def add_qa_pair(self, ans_id, ans_str, query_id, query_str):
            self.qa_pair.append(QA(ans_id=ans_id, ans_str=ans_str, query_id=query_id, query_str=query_str))

        def save(self, **kwargs):
            self.created_at = datetime.now()
            return super().save(**kwargs)

    def send_docs(qa_pairs):
        """adds document with qa pair to elastic index
            assumes that index fields correspond to template in src.elasticsearch.create_doc_index.py
        Args:
            full_text: full text of document containing answer
            qa_pairs: list of dictionaries with key:value='ans_id':integer, 'ans_str':str, 'ans_index'=integer, 'query_str'=str, 'query_id'=integer
        Returns: 
            document and qa_pair indexed to Elastic
        """
        print('uploading docs')
        counter = 0
        for pair in qa_pairs: 
            first = Doc(doc=pair['ans_str'])
            first.add_qa_pair(pair['ans_id'], pair['ans_str'], pair['query_id'], pair['query_str'])
            first.save()
            counter += 1 
        return counter

    qa_pairs = pd.read_csv(csv_file.file).fillna('nan').to_dict('records')
    counter = send_docs(qa_pairs)
    return {'message': 'success', 'number of docs': counter}