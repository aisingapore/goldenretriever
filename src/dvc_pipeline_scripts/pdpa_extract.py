from elasticsearch import Elasticsearch
import pandas as pd
import click 
import os


@click.command()
@click.option('--index_name', help='name of index to export as csv')
@click.option('--csv_prefix', help='prefix to add to -.csv of index')
@click.option('--savedir', help='path to save csv file')
def main(index_name, csv_prefix, savedir):
    print("executed")
    es = Elasticsearch()
    res = es.search(index=index_name, body={'query': {'match_all': {}}})
    print("got %d hits:" % res['hits']['total']['value'])

    df = pd.DataFrame(columns=['ans_id', 'ans_str', 'query_str', 'query_id'])
    for hit in res['hits']['hits']:
        tmp_df = pd.DataFrame(hit['_source']['qa_pair'])
        df = df.append(tmp_df, ignore_index=True)

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    save_path = os.path.join(savedir, csv_prefix + '.csv')
    df.to_csv(save_path)


if __name__=='__main__':
    main()