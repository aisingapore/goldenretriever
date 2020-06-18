""""""
"""
Version:
--------
0.1 11th May 2020

Usage:
------
Script to handle indexing of QnA datasets into Elasticsearch for downstream finetuning and serving
- Define index schema using elasticsearch_dsl classes
- Connect and upload Documents to Elasticsearch
"""
import pandas as pd 
import click

from datetime import datetime
from elasticsearch_dsl import Index, Document, InnerDoc, Date, Nested, Keyword, Text, Integer, connections
from argparse import ArgumentParser


@click.command()
@click.option('--url', help='elasticsearch url')
@click.option('--csv_file', help='csv file with qa pairs')
@click.option('--index_name', help='name of index to create')
def main(url, csv_file, index_name):

    index = Index(index_name)

    # index schema
    class QA(InnerDoc):
        ans_id = Integer()
        ans_str = Text(fields={'raw': Keyword()})
        query_id = Integer()
        query_str = Text()


    @index.document
    class Doc(Document):
        doc = Text()
        created_at = Date()
        qa_pair = Nested(QA)

        def add_qa_pair(self, ans_id, ans_str, query_id, query_str):
            self.qa_pair.append(QA(ans_id=ans_id, ans_str=ans_str, query_id=query_id, query_str=query_str))

        def save(self, **kwargs):
            self.created_at = datetime.now()
            return super().save(**kwargs)

    index.settings = {"number_of_shards": 1,
                      "number_of_replicas": 0}

    def upload_docs(qa_pairs):
        """
        adds document with qa pair to elastic index
        assumes that index fields correspond to template in create_doc_index.py

        :param full_text: full text of document containing answer
        :param qa_pairs: list of dictionaries with key:value='ans_id':integer, 'ans_str':str, 'query_str'=str, 'query_id'=integer
        :return: document and qa_pair indexed to Elastic
        """
        print('uploading docs')
        counter = 0
        for pair in qa_pairs: 
            first = Doc(doc=pair['ans_str'])
            first.add_qa_pair(pair['ans_id'], pair['ans_str'], pair['query_id'], pair['query_str'])
            first.save()
            counter += 1 
        print("indexing finished")
        print(f'indexed {counter} documents')

    # connect to ES instance and start indexing
    connections.create_connection(hosts=[url])
    qa_pairs = pd.read_csv(csv_file).fillna('nan').to_dict('records')
    counter = upload_docs(qa_pairs)


if __name__ == '__main__':
    main()

