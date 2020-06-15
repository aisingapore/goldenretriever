from os.path import join, dirname, abspath
from elasticsearch_dsl import Index, Document, Date, Integer, Text, Boolean
from datetime import datetime


PROJECT_ROOT = abspath(dirname(dirname(dirname(__file__))))

ES_URL = 'http://localhost:9200' # change this to your own ES URL Endpoint 

# change to your own qa index name 
QA_INDEX = 'qa_pdpa'

# name of index that stores log of user queries 
QUERY_LOG = 'querylog'

# paths for nearest neighbor index
INDEX_BUCKET = 'pdpa-index'
INDEX_PICKLE = 'pdpa-data.pkl'
INDEX_FILE = 'pdpa.idx'
INDEX_PREFIX = 'pdpa'
INDEX_FOLDER = join(PROJECT_ROOT, 'model_artefacts')
INDEX_PICKLE_PATH = join(INDEX_FOLDER, INDEX_PICKLE)
INDEX_FILE_PATH = join(INDEX_FOLDER, INDEX_FILE)

# path for dotenv file
DOTENV_PATH = join(PROJECT_ROOT, '.env')

# query log schema
class QueryLog(Document):
    created_at = Date()
    query_id = Text()
    query_text: Text()
    responses: Text(multi=True)  # allow multi responses in a List
    is_correct: Boolean(multi=True)
    feedback_timestamp: Date()

    class Index:
        name = QUERY_LOG

    def save(self, **kwargs):
        self.created_at = datetime.now()
        return super().save(**kwargs)
