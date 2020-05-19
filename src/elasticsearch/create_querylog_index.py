"""
Version:
--------
0.1 11th May 2020

Usage:
------
Script to store user queries
https://stackoverflow.com/questions/36650424/best-way-to-save-users-search-queries-in-elasticsearch

- Schema:
 "user_queries": [
        { id: 1
          text: "The query text",
          responses: "List of returned responses"
          is_correct: True, False, True // feedback on responses if present
        },
        { id: 2
          text: "Some other query text" // This one has no feedback
          responses: "List of returned response:
          is_correct: List of Booleans
        }
      ]
"""

from argparse import ArgumentParser
from datetime import datetime
from elasticsearch_dsl import Index, Boolean, Document, Date, Text, Integer, connections, Mapping


if __name__ == '__main__':
    parser = ArgumentParser(description='log user queries to Elasticsearch')
    parser.add_argument('url', default='localhost', help='elasticsearch_url')
    parser.add_argument('index_name', default='querylog', help='name of query log index')
    args = parser.parse_args()

    index = Index(args.index_name)

    index.settings = {"number_of_shards": 1, "number_of_replicas": 0}

    @index.document
    class QueryLog(Document):
        created_at = Date()
        query_id = Text()
        query_text: Text()
        responses: Text(multi=True)  # allow multi responses in a List
        is_correct: Boolean(multi=True)
        feedback_timestamp: Date()

        def save(self, **kwargs):
            self.created_at = datetime.now()
            return super().save(**kwargs)

    # bug where using Document class to create index seems to not be working and Mapping needs to be defined explicityly
    connections.create_connection(hosts=[args.url])
    QueryLog.init()
    print('querylog_index_created')
    m = Mapping()
    m.field('created_at', 'date')
    m.field('query_id', 'text')
    m.field('query_text', 'text')
    m.field('responses', 'text', multi=True)
    m.field('is_correct', 'boolean', multi=True)
    m.field('feedback_timestamp', 'date')
    m.save('querylog')
    ql = Index('querylog')
    print(ql.get_mapping())