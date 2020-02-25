"""
Flask app for GoldenRetriever
The script interfaces with a non-public db

The app may allow 2 methods:
    1. make_query
    2. save_feedback 

To use:
    python app_flask.py -db "db_cnxn_str.txt"
"""
import datetime
import pyodbc
import numpy as np
import pandas as pd
import pandas.io.sql as pds
from flask import Flask, jsonify, request
from waitress import serve
import argparse
import tarfile

from src.model import GoldenRetriever
from src.kb_handler import kb_handler




"""
Setup
"""
app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("-db", "--credentials", dest='dir',
                     default='db_cnxn_str.txt', 
                     help="directory of the pyodbc password string")
args = parser.parse_args()


class InvalidUsage(Exception):
    """
    Raises exception
    https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
    """
    status_code = 400

    def __init__(self, message="query endpoint requires arguments: query, kb_name", 
                 status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

def get_last_insert_ids(cursor, inserted_iterable = ['single_string']):
    """
    Get ids of last inserted iterable
    
    args:
    ----
        cursor: pyodbc cursor
        inserted_iterable: iterable of last inserted values
    
    Return:
    ------
        last_insert_ids: (list of ints) list of indices of inserted rows
    
    references:
        https://code.google.com/archive/p/pyodbc/wikis/FAQs.wiki#How_do_I_retrieve_autogenerated%2Fidentity_values%3F
        https://dba.stackexchange.com/questions/81604/how-to-insert-values-in-junction-table-for-many-to-many-relationships
        https://stackoverflow.com/questions/2548493/how-do-i-get-the-id-after-insert-into-mysql-database-with-python
    """
    cursor.execute( "SELECT @@IDENTITY")
    last_insert_id = cursor.fetchall()
    last_insert_id = int(last_insert_id[0][0])
    last_insert_ids = [i for i in range(last_insert_id, last_insert_id-len(inserted_iterable), -1)]
    return last_insert_ids

def extract(lst, idx=0): 
    return [item[idx] for item in lst] 

def init_sql_references(conn):
    """
    Utility function to get references from SQL. 
    The returned objects conveniently identify users based on kb_name or user hashkey
    """
    # get kb_names to kb_id
    kb_ref = pds.read_sql("""SELECT id, kb_name, directory_id  FROM dbo.kb_raw""", conn)
    get_kb_dir_id = kb_ref.loc[:,['kb_name', 'directory_id']].set_index('kb_name').to_dict()['directory_id']
    get_kb_raw_id = kb_ref.loc[:,['kb_name', 'id']].set_index('kb_name').to_dict()['id']

    # get kb permissions
    permissions = pds.read_sql("SELECT hashkey, kb_name, user_id FROM dbo.users \
                                LEFT JOIN dbo.kb_directory ON dbo.users.id = dbo.kb_directory.user_id \
                                LEFT JOIN kb_raw ON dbo.kb_directory.id = dbo.kb_raw.directory_id \
                            ", conn)
    permissions = pd.DataFrame(np.array(permissions), columns = ['hashkey', 'kb_name', 'user_id']).set_index('hashkey')
    
    return get_kb_dir_id, get_kb_raw_id, permissions





"""
Caching
------
Caches the following:
    1. gr: model object to make query
    2. cursor: SQL connection
    3. get_kb_dir_id, get_kb_raw_id: dictionary to retrieve
                                     kb_dir_id and kb_raw_id 
                                     from kb_name (user provided string)
"""
# load the model and knowledge bases
gr = GoldenRetriever()
# gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
kbh = kb_handler()
kbs = kbh.load_sql_kb(cnxn_path = args.dir, kb_names=['PDPA','nrf'])
gr.load_kb(kbs)


# make the SQL connection and cursor
conn = pyodbc.connect(open(args.dir, 'r').read())
cursor = conn.cursor()

get_kb_dir_id, get_kb_raw_id, permissions = init_sql_references(conn)




"""
API endpoints:
--------------
    1. make_query
    2. feedback 
    3. knowledge_base
"""
@app.route("/query", methods=['POST'])
def make_query():
    """
    Main function for User to make requests to. 

    Args:
    -----
        hashkey: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        query: (str) query string contains their natural question
        kb_name: (str) Name of knowledge base to query
        top_k: (int, default 5) Number of top responses to query. Currently kept at 5

    Return:
    -------
        reply: (list) contains top_k string responses
        query_id: (int) contains id of the request to be used for when they give feedback
    """

    # 1. parse the request and get timestamp
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()
    
    if not all([key in ['query', 'kb_name'] for key in request_dict.keys()]):
        raise InvalidUsage()

    HASHKEY = request_dict.get('hashkey')
    query_string = request_dict["query"]
    kb_name = request_dict["kb_name"]
    
    # # Manage KB access
    # try:
    #     if kb_name in permissions.loc[HASHKEY].kb_name:
    #         pass
    #     else:
    #         raise InvalidUsage(f"Unauthorised or unfound kb: {HASHKEY} tried to access {kb_name}")
    # except:
    #     raise InvalidUsage(f"Unrecognized hashkey: {HASHKEY}")



    # 2. model inference
    reply, reply_index = gr.make_query(query_string, 
                                       # top_k=int(top_k), 
                                       top_k = 5,
                                       index=True, kb_name=kb_name)


    # 3. log the request in SQL
    # query log has the following columns
    # id, created_at, query_string, user_id, kb_dir_id, kb_raw_id, Answer1, Answer2, Answer3, Answer4, Answer5
    rowinfo = [request_timestamp, query_string] 
    # append user_id
    logged_user_id = permissions.loc[HASHKEY].user_id.iloc[-1] if HASHKEY in permissions.index else None
    rowinfo.append(logged_user_id) 
    # append kb_dir_id
    rowinfo.append(get_kb_dir_id[kb_name])   
    # append kb_raw_id
    rowinfo.append(get_kb_raw_id[kb_name])
    # returned answers clause_id
    rowinfo.extend(gr.kb[kb_name].responses.clause_id.iloc[reply_index].tolist())

    cursor.execute('INSERT INTO query_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', rowinfo)
    cursor.commit()

    # 4. Return response to user
    # return id of latest log request to user for when they give feedback
    current_request_id = get_last_insert_ids(cursor)

    return jsonify(responses=reply, query_id=current_request_id)



@app.route("/feedback", methods=['POST'])
def save_feedback():
    """
    Retrieve feedback from end users

    args:
    ----
        query_id: (int) specifies the query to raise feedback for
        is_correct: (list) list fo booleans for true or false
    """
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()

    if not all([key in ['query_id', 'is_correct'] for key in request_dict.keys()]):
        raise InvalidUsage("request requires 'query_id', 'is_correct")

    # 1. parse the request
    query_id = request.get_json()["query_id"]
    is_correct = request.get_json()["is_correct"]
    is_correct = is_correct+[False]*(5-len(is_correct)) if len(is_correct) < 5 else is_correct # ensures 5 entries

    # log the request in SQL
    rowinfo = [request_timestamp]
    rowinfo.append(query_id)
    rowinfo.extend(is_correct[:5]) # ensures only 5 values are logged

    cursor.execute('INSERT INTO feedback_log VALUES (?, ?, ?, ?, ?, ?, ?)', rowinfo)
    cursor.commit()

    return jsonify(message="Success")



@app.route("/knowledge_base", methods=['POST'])
def upload_knowledge_base_to_sql():
    """
    Receive knowledge bases from users
    
    args:
    ----
        hashkey: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        kb_name: (str) Name of knowledge base to save as
        kb: (dict) contains the responses, queries and mappings
                   where mapping maps the indices of (question, answer)


    Sample json body & sample kb:
        {
         'hashkey': HASHKEY,
         'kb_name':'test1',
         'kb':{'responses': ["I'm 21 years old", 
                             "I hate mondays"],
               'contexts': ["Bob", "Gary"],
               'queries': ["What do you not love?", 
                           "How old are you?"],
               'mapping': [(0,1), (1,0)]
              }
        } 
    """
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()

    # verify that required arguments are inside
    if not all([key in ['hashkey','kb_name', 'kb'] for key in request_dict.keys()]):
        raise InvalidUsage(message="knowledge_base endpoint requires arguments: hashkey, kb_name, kb")

    HASHKEY = request_dict.get('hashkey', '')
    kb_name = request_dict["kb_name"]
    kb = request_dict["kb"]

    # get user id
    global get_kb_dir_id, get_kb_raw_id, permissions
    try:
        user_id = permissions.loc[HASHKEY].user_id.iloc[-1]
    except:
        print(f"ERROR: Tried to retrieve user id from {HASHKEY}")
        print( permissions.loc[HASHKEY] )
        raise InvalidUsage(message="Hashkey not recognized")



    """
    Load the knowledge base
    """
    # 1a. load to kb_directory
    cursor.execute('INSERT INTO dbo.kb_directory VALUES ( ?, ?, ?)', 
                    [request_timestamp, kb_name, user_id])
    cursor.commit()

    # 1b. get kb_directory index to load into kb_raw
    kb_dir = pds.read_sql("""
                        SELECT * from dbo.kb_directory 
                        WHERE user_id = (?)
                        AND dir_name = (?)
                        """,
                        conn,
                        params = [user_id, kb_name],
                        )
    kb_dir_idx = kb_dir.id.iloc[-1]



    # 2a. load into kb_raw
    # https://stackoverflow.com/questions/41973933/invalid-parameter-type-numpy-int64-when-inserting-rows-with-executemany
    # kb_dir_idx has to be integer typed
    cursor.execute('INSERT INTO dbo.kb_raw VALUES ( ?, ?, ?, ?)', 
                    [None, kb_name, 'user_uploaded', int(kb_dir_idx)])
    cursor.commit()

    # 2b. update SQL references and get kb_raw id
    get_kb_dir_id, get_kb_raw_id, permissions = init_sql_references(conn)
    kb_raw_dir_idx = get_kb_raw_id[kb_name]



    # 3. load into kb_clauses
    responses = kb['responses']
    
    context_strings = kb.get('contexts', [])
    context_strings = ['']*len(responses) if len(context_strings)==0 else context_strings
    if len(context_strings) != len(responses):
        raise InvalidUsage(message="contexts should either have the same number of strings as responses or excluded")

    list_of_clauses = [[kb_raw_dir_idx, clause_ind, context_string, raw_string, context_string + '\n' + raw_string, request_timestamp]
                        for clause_ind, (context_string, raw_string) in enumerate(zip(context_strings, responses))
                        ]

    cursor.executemany('INSERT INTO dbo.kb_clauses VALUES ( ?, ?, ?, ?, ?, ?)', 
                        list_of_clauses)
    cursor.commit()
    idx_of_inserted_clauses = get_last_insert_ids(cursor, list_of_clauses)
    
    print(kb.keys())
    if all(key_ in kb.keys() for key_ in ['queries', 'mapping']):
        if (len(kb['queries'])>0) & (len(kb['mapping'])>0):

            print("loading labels")

            # 4. load into query_db
            cursor.executemany('INSERT INTO dbo.query_db VALUES (?)', 
                                [[query_] for query_ in kb['queries']])
            cursor.commit()
            idx_of_inserted_queries = get_last_insert_ids(cursor, kb['queries'])
    

            # 5. load into query_labels
            #    query labels have the following columns
            #    query_id, clause_id, span_start, span_end, created_at
            mapped_query_ids = pd.Series(idx_of_inserted_queries).iloc[extract(kb['mapping'], idx=0)]
            mapped_clause_ids = pd.Series(idx_of_inserted_clauses).iloc[extract(kb['mapping'], idx=1)]

            list_of_query_labels = [[mapped_query_id, mapped_clause_id , None, None, request_timestamp]
                                    for mapped_query_id, mapped_clause_id
                                    in zip(mapped_query_ids,mapped_clause_ids )
                                    ]

            cursor.executemany('INSERT INTO dbo.query_labels VALUES (?, ?, ?, ?, ?)', 
                                list_of_query_labels)
            cursor.commit()


    return jsonify(message="Success")




@app.route("/delete", methods=['POST'])
def remove_knowledge_base_from_sql():
    """
    Remove knowledge bases from SQL database
    
    args:
    ----
        hashkey: (str, optional) identification; intended to be their hashkey 
                                 to manage exclusive knowledge base access.
        kb_name: (str) Name of knowledge base to save as


    Sample json body & sample kb:
        {
         'hashkey': HASHKEY,
         'kb_name':'test1',
        } 
    """
    request_timestamp = datetime.datetime.now()
    request_dict = request.get_json()

    # verify that required arguments are inside
    if not all([key in ['hashkey','kb_name'] for key in request_dict.keys()]):
        raise InvalidUsage(message="delete endpoint requires arguments: hashkey, kb_name")

    HASHKEY = request_dict.get('hashkey', '')
    kb_name = request_dict["kb_name"]

    del_kb = pds.read_sql('''SELECT dbo.kb_raw.kb_name, dbo.kb_clauses.processed_string, dbo.query_db.query_string, dbo.kb_clauses.created_at, \
                            dbo.kb_raw.directory_id, dbo.kb_clauses.raw_id, dbo.query_labels.clause_id, dbo.query_labels.query_id, dbo.query_labels.id \
                            FROM dbo.users \
                            LEFT JOIN dbo.kb_directory ON dbo.users.id = dbo.kb_directory.user_id \
                            LEFT JOIN dbo.kb_raw ON dbo.kb_directory.id = dbo.kb_raw.directory_id \
                            LEFT JOIN dbo.kb_clauses ON dbo.kb_raw.id = dbo.kb_clauses.raw_id \
                            LEFT JOIN dbo.query_labels ON dbo.kb_clauses.id = dbo.query_labels.clause_id \
                            LEFT JOIN dbo.query_db ON dbo.query_labels.query_id = dbo.query_db.id \
                            WHERE dbo.kb_raw.kb_name = (?) \
                            AND dbo.users.hashkey = (?) \
                            AND dbo.kb_raw.type = 'user_uploaded'
                            ''',
                        conn,
                        params = [kb_name, HASHKEY],
                        )
    del_kb.rename({'id':'query_labels_id'}, axis=1, inplace=True)   

    if len(del_kb)==0:
        return jsonify(message=f"No entries from kb_name {kb_name} to delete")

    column_to_table = {
        'query_labels_id':'query_labels',
        'query_id':'query_db', 
        'clause_id':'kb_clauses', 
        'raw_id':'kb_raw', 
        'directory_id':'kb_directory', 
    }

    for id_column, table_name in column_to_table.items():

        id_to_delete = del_kb.loc[:,id_column].dropna().apply(int).unique().tolist()
        id_to_delete = [[id_] for id_ in id_to_delete]
        print(f"{table_name} \n {id_to_delete} \n")

        if len(id_to_delete)>0:
            cursor.executemany('DELETE FROM "{}" WHERE id = (?)'.format(table_name), id_to_delete)
            
    cursor.commit()

    return jsonify(message="Success")


@app.route("/upload_weights", methods=['POST'])
    """
    Upload finetuned weights to microsoft azure blob storage container
    
    args:
    ----
        access_key: (str) key for authorized access to blob storage container.
        container_name: (str) name of newly created container that will store weights
        weights_folder_name: (str) name of the newly stored weights .gz file in the container
        weights_path: (str) path leading to the weights that will be uploaded to the container


    Sample json body:
        {
         'access_key': ACCESSKEY,
         'container_name': CONTAINER_NAME,
         'weights_folder_name': WEIGHTS_FOLDER_NAME
         'weights_path': WEIGHTS_PATH
        } 
    """

    if not all([key in ['access_key', 'container_name', 'weights_folder_name', 'weights_path'] for key in request_dict.keys()]):
        raise InvalidUsage(message="upload_weights endpoint requires arguments: access_key, container_name, weights_folder_name, weights_path")

    ACCESSKEY = request_dict.get('access_key', '')
    CONTAINER_NAME = request_dict.get('container_name', '')
    WEIGHTS_FOLDER_NAME = request_dict.get('weights_folder_name', '')
    WEIGHTS_PATH = request.get('weights_path','')

    # Create the BlobServiceClient that is used to call the Blob service for the storage account
    conn_str = ACCESSKEY
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

    # Create a container called 'quickstartblobs' and Set the permission so the blobs are public.
    blob_service_client.create_container(
        CONTAINER_NAME, public_access=PublicAccess.Container)

    with tarfile.open(WEIGHTS_FOLDER_NAME, "w:gz") as tar:
        tar.add(WEIGHTS_PATH, arcname=os.path.basename(WEIGHTS_FOLDER_NAME))

    ZIP_FOLDER_PATH = WEIGHTS_PATH + WEIGHTS_FOLDER_NAME

    print("\nUploading to Blob storage as blob" + ZIP_FOLDER_PATH)

    # Upload the created file, use local_file_name for the blob name
    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME, blob=WEIGHTS_FOLDER_NAME)

    blob_client.upload_blob(ZIP_FOLDER_PATH)

    # List the blobs in the container
    container = blob_service_client.get_container_client(container=CONTAINER_NAME)
    generator = container.list_blobs()
    all_blobs = ["\t Blob name: " + blob.name for blob in generator]

    return jsonify(message=all_blobs)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port="5000")
    serve(app, host='0.0.0.0', port=5000, url_scheme='https')