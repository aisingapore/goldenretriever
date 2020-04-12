import sqlite3
import pyodbc
import datetime
import pandas as pd

from app.api.upload_kb_service import upload_knowledge_base_to_sql


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(e)

def create_connection(conn_path):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(conn_path, check_same_thread=False)
        print(sqlite3.version)
    except Exception as e:
        print(e)
    return conn

def create_test_db(conn_path):

    conn = create_connection(conn_path)
    cursor = conn.cursor()

    sql_statement = "ATTACH '%s' as dbo;" % conn_path
    cursor.execute(sql_statement)

    sql_create_kb_clauses = """CREATE TABLE IF NOT EXISTS dbo.kb_clauses (
        id INTEGER PRIMARY KEY,
        raw_id INTEGER,
        clause_ind INTEGER,
        context_string varchar,
        raw_string varchar,
        processed_string varchar,
        created_at DATETIME
    );"""

    sql_create_query_db = """CREATE TABLE IF NOT EXISTS dbo.query_db (
        id INTEGER PRIMARY KEY,
        query_string varchar NOT NULL
    );"""

    sql_create_query_labels = """CREATE TABLE IF NOT EXISTS dbo.query_labels (
        id INTEGER PRIMARY KEY,
        query_id INTEGER,
        clause_id INTEGER,
        span_start INTEGER,
        span_end INTEGER,
        created_at DATETIME
    );"""

    sql_create_kb_raw = """CREATE TABLE IF NOT EXISTS dbo.kb_raw (
        id INTEGER PRIMARY KEY,
        filepath varchar,
        kb_name varchar NOT NULL,
        type varchar,
        directory_id int
    );"""    

    sql_create_kb_directory = """CREATE TABLE IF NOT EXISTS dbo.kb_directory (
        id INTEGER PRIMARY KEY,
        created_at DATETIME,
        dir_name varchar NOT NULL,
        user_id int
    );"""

    sql_create_query_log = """CREATE TABLE IF NOT EXISTS dbo.query_log (
        id INTEGER PRIMARY KEY,
        created_at DATETIME,
        query_string varchar,
        user_id int,
        kb_dir_id int,
        kb_raw_id int,
        Answer1 int,
        Answer2 int,
        Answer3 int,
        Answer4 int,
        Answer5 int
    );"""

    sql_create_users = """CREATE TABLE IF NOT EXISTS dbo.users (
        id INTEGER PRIMARY KEY,
        created_at DATETIME,
        email varchar(50) NOT NULL UNIQUE,
        full_name varchar,
        org_name varchar,
        hashkey varchar
    );"""

    sql_create_feedback_log = """CREATE TABLE IF NOT EXISTS dbo.feedback_log (
        id INTEGER PRIMARY KEY,
        created_at DATETIME,
        query_log_id int,
        feedback1 BIT,
        feedback2 BIT,
        feedback3 BIT,
        feedback4 BIT,
        feedback5 BIT
    );"""

    if conn:
        # create tables
        create_table(conn, sql_create_users)
        create_table(conn, sql_create_kb_directory)
        create_table(conn, sql_create_kb_raw)
        create_table(conn, sql_create_query_log)
        create_table(conn, sql_create_kb_clauses)
        create_table(conn, sql_create_query_db)
        create_table(conn, sql_create_query_labels)
        create_table(conn, sql_create_feedback_log)
        conn.commit()

    else:
        print('Error! cannot create db connection')

    return conn

class MockRequest:
    def __init__(self, hashkey=None, kb_name=None, kb=None):
        self.hashkey = hashkey
        self.kb_name = kb_name
        self.kb = kb


def insert_data_into_test_db(conn):

    get_kb_dir_id = {'nrf': 1}
    get_kb_raw_id = {'nrf': 1}
    permissions = pd.DataFrame([["12345", "nrf", 1], ["12345", 'PDPA', 1]],
                               columns=['hashkey', 'kb_name', 'user_id']).set_index('hashkey')

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM dbo.kb_clauses")

    data = cursor.fetchall()

    if data:
        pass
    else:
        # Upload mock nrf data into test_db
        d = {"responses": ["Periodic Audit Report 34...",
                        "Change in Research Scope 32...",
                        "Yearly Progress Report 35...",
                        "Acknowledgement Guidelines 40...",
                        "Grant Extension 27..."],
            "contexts": [],
            "queries": ["Can we seek for an extension of the audit report submission deadline?",
                        "What should we do if there are changes to the research project",
                        "What is the deadline for the submission of the annual progress report",
                        "We would like to reflect 'Endorsed by Company A' in our advertisement posters, will that be ok?",
                        "Are we able to extend our project deadline?"],
            "mapping": [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}

        mock_req = MockRequest(hashkey="12345", kb_name="nrf", kb=d)

        upload_knowledge_base_to_sql(mock_req, conn, cursor, get_kb_dir_id,
                                    get_kb_raw_id, permissions)

        # Populate mock dbo.users table
        current_dt = datetime.datetime.now()

        cursor.execute('INSERT INTO dbo.users (created_at, email, full_name, org_name, hashkey) VALUES ( ?, ?, ?, ?, ?)',
                    [current_dt, "test_user@aisingapore.org", "test_user", "aisg", "12345"])

        conn.commit()

    return get_kb_dir_id, get_kb_raw_id, permissions
