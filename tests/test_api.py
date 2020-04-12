import os
import pytest
import tarfile

from azure.storage.blob import BlobServiceClient
from fastapi.testclient import TestClient

from app.api.main import app, get_common_params
from src.models import GoldenRetriever
from src.data_handler.kb_handler import kb_handler
from tests.setup_test_env import create_test_db, insert_data_into_test_db


def override_get_common_params():
    conn_path = './goldenretriever.db'
    
    conn = create_test_db(conn_path)
    cursor = conn.cursor()

    # Mock data
    get_kb_dir_id, get_kb_raw_id, permissions = insert_data_into_test_db(conn)

    kbh = kb_handler()
    kbs = kbh.load_sql_kb(cnxn_path=conn_path, kb_names=['nrf'])

    gr = GoldenRetriever()
    # gr.restore('./google_use_nrf_pdpa_tuned/variables-0')
    gr.load_kb(kbs)

    return {"conn": conn, "conn_path": conn_path, "get_kb_dir_id": get_kb_dir_id, "get_kb_raw_id": get_kb_raw_id, "permissions": permissions, "kbs": kbs, "gr": gr, "kbh": kbh}


app.dependency_overrides[get_common_params] = override_get_common_params

client = TestClient(app)
DB_HASH_KEY = os.environ['DB_HASH_KEY']
AZURE_STORAGE_CONN_STR = os.environ['AZURE_STORAGE_CONN_STR']
CONTAINER_NAME = "testweights"

query_id = None


def test_query():
    d = {
         "hashkey": DB_HASH_KEY,
         "query": "Can I change funding source",
         "kb_name": "nrf"
    }

    response = client.post(
        "/query",
        json=d)

    data = response.json()

    global query_id
    query_id = data["query_id"][0]

    assert isinstance(query_id, int)
    assert isinstance(data["responses"][0], str)
    assert response.status_code == 200


def test_feedback():

    global query_id

    d = {
         "query_id": query_id,
         "is_correct": [1, 0, 0, 0, 0]
    }

    response = client.post(
        "/feedback",
        json=d)

    data = response.json()

    assert data["message"] == "Success"
    assert response.status_code == 200


def test_knowledge_base():
    d = {"hashkey": DB_HASH_KEY,
         "kb_name": "Test_data",
         "kb": {"responses": ["I'm 21 years old", "I hate mondays"],
                "contexts": ["Bob", "Gary"],
                "queries": ["What do you not love?", "How old are you?"],
                "mapping": [(0, 1), (1, 0)]}
    }

    response = client.post(
        "/knowledge_base",
        json=d)

    data = response.json()

    assert data["message"] == "Success"
    assert response.status_code == 200


@pytest.fixture
def create_dummy_weights():
    
    os.makedirs("./test_folder")

    with open("./test_folder/test.txt", "w") as file:
        file.write("dummy txt here")
    
    with tarfile.open("./weights.tar.gz", "w:gz") as tar:
        tar.add("./test_folder")

    tar_path = os.path.join(os.getcwd(), "weights.tar.gz")
    yield tar_path

    # Remove weights in test environment
    os.remove("./test_folder/test.txt")
    os.remove("weights.tar.gz")

    # Remove container on Microsoft Azure
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=AZURE_STORAGE_CONN_STR)
    blob_service_client.delete_container(CONTAINER_NAME)



def test_upload_weights(create_dummy_weights):
    d = {
         "conn_str": AZURE_STORAGE_CONN_STR,
         'container_name': CONTAINER_NAME,
         'blob_name': "weights.tar.gz"}

    tar_path = create_dummy_weights

    files = {'file': open(tar_path, 'rb')}

    response = client.post(
        "/upload_weights",
        data=d,
        files=files)

    data = response.json()

    assert data["message"] == "Success"
    assert response.status_code == 200
