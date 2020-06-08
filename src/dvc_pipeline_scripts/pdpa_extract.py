from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd 

es = Elasticsearch()

res = scan(es, query={'query': {'match_all': {}}}, index='pdpa-qa')  # returns generator object
res_ls = list(res)
print(f'scan query returns {len(res_ls)} documents')
df = pd.DataFrame(columns=['ans_id', 'ans_str', 'query_str', 'query_id'])
for hit in res_ls:
    tmp_df = pd.DataFrame(hit['_source']['qa_pair'])
    df = df.append(tmp_df, ignore_index=True)
print(f'final df has dimensions {df.shape}')
print(df.head())
df.to_csv('~/Desktop/golden-retriever/data/pdpa_test.csv')
print(f'saved data to csv')