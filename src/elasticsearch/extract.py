from elasticsearch import Elasticsearch
import pandas as pd 

es = Elasticsearch()

res = es.search(index='qa_nrf', body={'query': {'match_all': {}}})
print("got %d hits:" % res['hits']['total']['value'])
df = pd.DataFrame(columns=['ans_id', 'ans_str', 'query_str', 'query_id'])
for hit in res['hits']['hits']:
    tmp_df = pd.DataFrame(hit['_source']['qa_pair'])
    df = df.append(tmp_df, ignore_index=True)
print(df.columns)
print(df.head)
df.to_csv('~/Desktop/golden-retriever/data/nrf_test.csv')