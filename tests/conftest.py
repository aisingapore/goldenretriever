import os 
from elasticsearch.helpers.test import SkipTest, get_test_client
from elasticsearch_dsl.connections import connections, add_connection
from mock import Mock 
import pytest
from pytest import fixture, skip
from starlette.testclient import TestClient
from app.api.main_es import app
import tarfile


@pytest.fixture(scope='module')
def test_app():
      client = TestClient(app)
      yield client

@fixture(scope='session')
def es_client():
    try:
        connection = get_test_client(nowait='WAIT_FOR_ES' not in os.environ)
        add_connection('default', connection)
        return connection
    except SkipTest:
        skip()

@pytest.fixture
def create_dummy_weights(tmpdir): 
    d = tmpdir.mkdir('test_folder')

    with open(d + "/test.txt", "w") as file:
        file.write("dummy txt here")
  
    with tarfile.open("./weights.tar.gz", "w:gz") as tar:
        tar.add(d)

    tar_path = os.path.join(os.getcwd(), "weights.tar.gz")
    yield tar_path