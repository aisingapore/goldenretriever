from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd 
import click 
import os


@click.command()
@click.option('--index_name', help='name of index to export as csv')
@click.option('--csv_prefix', help='prefix to add to -.csv of index')
@click.option('--savedir', help='path to save csv file')
def main(index_name, csv_prefix, savedir):
    es = Elasticsearch()
    
    res = scan(es, query={'query': {'match_all': {}}}, index=index_name)  # returns generator object
    res_ls = list(res)
    print(f'scan query returns {len(res_ls)} documents')
    
    df = pd.DataFrame(columns=['ans_id', 'ans_str', 'query_str', 'query_id'])
    for hit in res_ls:
        tmp_df = pd.DataFrame(hit['_source']['qa_pair'])
        df = df.append(tmp_df, ignore_index=True)
    print(f'final df has dimensions {df.shape}')
    print(df.head())

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    save_path = os.path.join(savedir, csv_prefix + '.csv')
    df.to_csv(save_path)
    print(f'saved data to csv')
    
if __name__=='__main__':
    main()
