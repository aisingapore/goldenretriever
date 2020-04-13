import tarfile

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, PublicAccess
from azure.core.exceptions import ResourceExistsError
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.api.db_handler import get_last_insert_ids, extract_qa_pair_based_on_idx, get_kb_id_ref, get_permissions, ensure_connection
from app.api.exceptions import InvalidUsage

def upload_weights(request, files=None):
    """
    Upload finetuned weights to an azure blob storage container
    
    args:
    ----
        conn_str: (str) connection string for authorized access to blob storage
        container_name: (str) name of newly created container that will store weights
        blob_name: (str) name to be used for the newly stored weights in the container


    Sample json body:
        {
         'conn_str': CONN_STR,
         'container_name': CONTAINER_NAME,
         'blob_name': BLOB_NAME
        } 
    """

    CONN_STR = request['conn_str']
    CONTAINER_NAME = request['container_name']
    BLOB_NAME = request['blob_name']

    # Create the BlobServiceClient that is used to call the Blob service for the storage account
    blob_service_client = BlobServiceClient.from_connection_string(conn_str=CONN_STR)

    # Create a container. Use public_access=PublicAccess.Container if container is open to public
    try:
        blob_service_client.create_container(CONTAINER_NAME, public_access=None)
    except ResourceExistsError as e:
        return 'Container already exists, please select another container_name and try again'
    except Exception as e:
        return str(e)

    # Upload file
    if files is not None:

        # Upload the created file, use WEIGHTS_FOLDER_NAME as the blob name
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=BLOB_NAME)

        blob_client.upload_blob(files)
    
    return "Success"