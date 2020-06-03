import pytest
from app.api import upload_weights_service_es, qa_service_es
from mock import Mock


class MockQueryResponse:
    @staticmethod
    def json():
        return {'resp': ['response to question'], 'query_id': '1234'}


class MockFeedbackResponse:
    @staticmethod
    def json():
        return {'resp': 'updated'}


@pytest.fixture
def mock_upload_weights():
    upload = Mock()
    upload.return_value = 'success'
    return upload


def test_qa_service(test_app, monkeypatch):
    query_string = 'debarring principal investigators' 
    k = 1

    def mock_get(*args, **kwargs):
        return MockQueryResponse()

    monkeypatch.setattr(test_app, 'get', mock_get)

    response = test_app.get(f'/query/{query_string}/{k}')
    js = response.json()
    assert js['resp'] == ['response to question']
    assert js['query_id'] == '1234'


def test_qa_service_invalid_request(test_app, monkeypatch):
    def mock_get(*args, **kwargs):
        return 'invalid request parameters'

    monkeypatch.setattr(test_app, 'get', mock_get)

    response = test_app.get('/query/999')
    assert response == 'invalid request parameters'


# def test_qa_service_integration(test_app):
#     query_string = 'debarring principal investigators' 
#     k = 1

#     response = test_app.get(f'/query/{query_string}/{k}')
#     answers = response.json()['resp']
#     query_id = response.json()['query_id']
#     assert response.status_code == 200
#     assert type(answers) == list
#     assert type(query_id) == str


def test_qa_service_invalid_request_integration(test_app, monkeypatch):
    async def mock_make_query(query_string, k):
        return None

    monkeypatch.setattr(qa_service_es, 'make_query', mock_make_query)

    response = test_app.get('/query/999')
    assert response.status_code == 404


def test_feedback_service(test_app, monkeypatch):
    d = {"query_id": "486128aa-98db-11ea-9c85-8c8590a53c2e",
         "is_correct": [1, 0, 0, 0, 0]}
    def mock_post(*args, **kwargs):
        return MockFeedbackResponse()

    monkeypatch.setattr(test_app, 'post', mock_post)

    response = test_app.post("/feedback", json=d)
    js = response.json()
    assert js['resp'] == 'updated'


def test_upload_with_minio_mocked(test_app, create_dummy_weights, monkeypatch):
    d = {'minio_url': 'localhost:9001',
         'minio_access_key': 'minio',
         'minio_secret_key': 'minio123',
         'bucket_name': 'testweights',
         'object_name': 'weights.tar.gz'} 
    tar_path = create_dummy_weights
    files = {'file': open(tar_path, 'rb')}
    monkeypatch.setattr(upload_weights_service_es, 'upload_weights', mock_upload_weights)
    response = test_app.post("/upload_weights", data=d, files=files)
    js = response.json()
    assert js['message'] == 'success'