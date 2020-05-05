from datetime import datetime
from elasticsearch_dsl import Index, Document, InnerDoc, Date, Nested, Keyword, Text, Integer, connections
import pandas as pd 

qa_nrf = Index('qa_nrf')

qa_nrf.settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }

class QA(InnerDoc):
    ans_id = Integer()
    ans_str = Text(fields={'raw': Keyword()})
    query_id = Integer()
    query_str = Text()

@qa_nrf.document
class Doc(Document):
    doc = Text()
    created_at = Date()
    qa_pair = Nested(QA)
    def add_qa_pair(self, ans_id, ans_str, query_id, query_str):
        self.qa_pair.append(QA(ans_id=ans_id, ans_str=ans_str, query_id=query_id, query_str=query_str))

    def save(self, **kwargs):
        self.created_at = datetime.now()
        return super().save(**kwargs)


def setup():
    """create an IndexTemplate and save it into elasticsearch"""
    index_template = Doc._index.as_template('base')
    index_template.save()


def upload_single_doc(qa_pairs):
    """adds document with qa pair to elastic index
        assumes that index fields correspond to template in create_doc_index.py
    Args:
        full_text: full text of document containing answer
        qa_pairs: list of dictionaries with key:value='ans_id':integer, 'ans_str':str, 'ans_index'=integer, 'query_str'=str, 'query_id'=integer
    Returns: 
        document and qa_pair indexed to Elastic
    """
    for pair in qa_pairs: 
        first = Doc(doc=pair['ans_str'])
        first.add_qa_pair(pair['ans_id'], pair['ans_str'], pair['query_id'], pair['query_str'])
        first.save()
    print("indexing finished")



if __name__=='__main__':
    DATA_FILEPATH = 'data/nrf.csv'
    connections.create_connection(hosts=['localhost'])

    # read data 
    nrf_df = pd.read_csv(DATA_FILEPATH).fillna('nan')
    nrf_js = nrf_df.to_dict('records')
    print('uploading docs')
    upload_single_doc(nrf_js)
    print('docs uploaded')

    

