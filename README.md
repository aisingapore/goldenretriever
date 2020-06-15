<h1 align="center"> Golden Retriever </h1>

_Setting up your own information retrieval tool has never been easier_

**Documentation**: [URL]() `TO BE UPDATED`

Retrieving information from your mobile phone or laptop seems to be a simple task. Similar to our experiences with Google’s search engine, all we need is to type our queries into a search bar, and in return we are able to receive a list of relevant responses. 

With Golden Retriever, we aim to provide an information retrieval engine for users across different industries to search through documents of their own in a similar fashion. We often have valuable information embedded within the text documents on our computer. For companies, these text documents could be in the form of Terms and Conditions, Frequently Asked Questions or contractual documents.

We are utilising the following open source tools to power the core of Golden Retriver,
- **[Google's Universal Sentence Encoder](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html?m=1)**: a model pre-trained by Google and used within Golden Retriever to encode text into high dimensional vectors for semantic similarity tasks.
- **[Elasticsearch](https://www.elastic.co/guide/index.html)**: a distributed RESTful search and analytics engine used to store the incoming queries and potential responses for your application
- **[Minio](https://min.io/)**: an object storage system used to store the fine-tuned weights of your model
- **[Streamlit](https://www.streamlit.io/)**: an easy-to-use package to setup a frontend page for users to send in their queries
- **[FastAPI](https://fastapi.tiangolo.com/)**: a web framework for building APIs with python 3.6+
- **[DVC](https://dvc.org/doc)**: DVC runs on top of any Git repository and allows users to setup reproducible Machine Learning pipelines

<h1 align="center"> Installation and requirements </h1>

**Python**
- 3.6 and above

**Docker**
- If you’re not sure if you have Docker installed, you can check by running:
```
docker -v && docker-compose -v

# Docker version 19.03.5, build 633a0ea
# docker-compose version 1.24.1, build 4667896b
```

If Docker is installed on your machine, you should see the output illustrating the installed version of Docker and Docker Compose. If you need to install docker, you can refer to the [Docker official website](https://docs.docker.com/get-docker/) for more information.

**Virtual environment**

- Setup a python virtual environment
```
virtualenv env
```

- Install the required dependencies 
```
pip install -r requirements.txt
```


<h1 align="center"> Folder directory </h1>

```
golden-retriever
|_app
  |_streamlit
  |_api
|_ data
|_ dvc
|_ notebooks
|_ src
|_ docker-compose.yml
|_ requirements.txt
|_ dvc_pipelines.sh

```
- app/streamlit: user-facing Streamlit app 
- app/api: FastAPI endpoints that will be used by the Streamlit app to process users' queries and return the relevant responses
- data: sample data
- dvc: dvc metafiles
- notebooks: notebooks for demonstration purposes
- src: refer to [documentation]()
- docker-compose.yml: YAML compose file defining docker images, services, networks and volumes. Will be used to spin up the multi-container Golden Retriever application
- requirements.txt: required python packaages to be installed
- dvc_pipelines.sh: optional script to setup dvc pipelines and metafiles

<h1 align="center"> Getting started </h1>

We will illustrate the process of using Golden Retriever by using the PDPA dataset.

1. Amend setup configuration in `src/finetune/config.py`

- If you plan to finetune your model, edit the path to save weights
```
config_dict = {"save_dir":"./test_finetune"}  #default is ./test_finetune
```
- You may choose to change the other finetuning variables too

2. Amend setup configuration in `app/api/config.py`. 

- **Elasticsearch endpoint** Specify your desired ES endpoint
```
ES_URL = 'http://localhost:9200' # change this to your own ES URL Endpoint 
```
- **Elasticsearch Index** Specify the name of your elasticsearch index. This is where you store your Questions and Answers 
```
QA_INDEX = 'qa_pdpa'. # change to your own qa index name 
```
- **Simple Nearest Neighbors Index** Golden Retriever precomputes the vector representations of the answers in your dataset and saves them into a Simple Nearest Neighbors index for lookup during serving time. You can set the names of the variables related to the index: 

```
# paths for nearest neighbor index
INDEX_BUCKET = 'pdpa-index'. # name of minio bucket to save index to
INDEX_PICKLE = 'pdpa-data.pkl'. # name of pickle file output by saving process
INDEX_FILE = 'pdpa.idx'. # name of .idx file output by saving process 
INDEX_PREFIX = 'pdpa'. # prefix used when loading index. This should corresponse to 'prefix-data.pkl' and 'prefix.idx'
```

3. Create a `.env` file in the project root folder to manage Minio access keys.
- **Minio configuration** Golden Retriever sets up Minio as an object store for model weights and other application binaries.  Here is a sample: 

```
# variables for handling minio
MINIO_URL=localhost:9001
ACCESS_KEY=minio # change this to a proper access key! 
SECRET_KEY=minio123 # change to to a proper secret key! 
```

2. Docker Compose makes it easy to run the following Docker container applications to form a fully integrated application. Build and start the multi-container application by typing:
```
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up
```

4. Uploading training/evaluation data to elasticsearch
- We will start by uploading the data that will be used to train and evaluate the model. You may upload the data in the form of a .csv file and it should contain 4 columns where each row should contain a query (query_str), a query id (query_id), the correct answer for the query (ans_str) and a answer id (ans_id). The ids for the queries should be unique ids for the query strings, and this is should be the case for the answers too. The ids, in particular for the queries, are used during the train-test split step to ensure that the same query does not appear in the training and evaluation phase. For multi-response queries, the queries should be repeated and each repetition should be tagged with a different clause. Refer to sample_multi_clause.csv and sample_single_clause.csv for the sample files.

- Upload the data to elasticsearch using the following command

```
--url: elasticsearch url
--csv_file: csv file with qa pairs
--index_name: name of index to create


python -m src.elasticsearch.create_doc_index --url localhost --csv_file data/pdpa.csv --index_name qa-pdpa
```

5. Finetune
- Do ensure that you have amended src/finetune/config.py to insert your preferred finetuning parameters and folder paths in before running the command below
```
python -m src.finetune.main
``` 

The finetuned weights and results will be found in the folder you have set in src/finetune/confg.py. For the evaluation step, we are using **Mean Reciprocal Rank (MRR)** metric, which is used for systems that return a ranked list of answers to queries. For a single query, the reciprocal rank is $`\frac{1}{rank}`$ where rank is the position of correct response. More information on MRR can be found [here](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 

6. Encode and save index
- You may use the .csv file in step 3 or alternatively use the src/dvc_pipeline_scripts/pdpa_extract.py file to retrieve the query and response strings from elasticsearch.
```
--index_name: name of index to export as csv
--csv_prefix: prefix to add to -.csv of index
--savedir: path to save csv file

python -m src.dvc_pipeline_scripts.index_extract --index_name pdpa --csv_prefix pdpa --savedir ./test
```



```
--data: path to file with raw responses
--output_folder: path to save index
--index_prefix: prefix to add to -data.pkl of index
--gr_model: name of gr model
--savedir: path to fine-tuned encoder weights

python -m src.dvc_pipeline_scripts.index_encode --data ./data/pdpa.csv --output_folder model_artefacts --index_prefix pdpa --gr_model USEEncoder --savedir ./finetune
```

7. Set up backend APIs (Fastapi) and frontend (Streamlit) apps
- Do ensure that you have completed step 2 before running the commands below

**Backend**

Run the app inside the virtualenv that you have created under *Installation and Requirements* section.
```
uvicorn app.api.main_es:app 
```

We include a docker image for running the FastAPI application. Note: the Simple Neighbors index sometimes get corrupted when being copied from the host folder into the Docker container, which may result in a `_pickle.UnpicklingError`. If this happens, use virtualenv instead. 
```
# Build the docker image
docker build -f app/api/api.Dockerfile -t goldenretriever_fastapi .

# Start the docker container
docker run -p 8000:80 -it goldenretriever_fastapi
```

if you are running it on your own computer, you can access the endpoints at http://localhost:8000

**Frontend**

You may start the streamlit directly by running the following command with the url for the API endpoints in the previous step and you should be able to access the streamlit application at http://localhost:8501
```
streamlit run app/streamlit/main.py -- --url <url for API endpoints>

eg. streamlit run app/streamlit/main.py -- --url http://localhost:8000
```


<h1 align="center"> Utilizing your own encoders </h1>
If you are interested in using other types of encoders apart from Google Universal Sentence Encoder, your encoder needs to inherit from the `Encoder` abstract class in src/encoders.py

```
class Encoder(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def finetune_weights(self):
        pass

    @abstractmethod
    def save_weights(self):
        pass

    @abstractmethod
    def restore_weights(self):
        pass
```

<h1 align="center"> Dvc pipelines (optional) </h1>

DVC allows versioning data files, model results, and ML models using Git, but without storing the file contents in the Git repository. DVC saves information about your data in special DVC-files, and these metafiles can be used for versioning. We recommend using DVC to track your Golden Retriever QA datasets and models for easy reproducibility.

The official DVC [tutorial](https://dvc.org/doc/tutorials) is a good place to start to setup DVC and to have a basic understanding of DVC features.

Generally a DVC pipeline contains a series of commands that take an input and produce an output. After creating the pipeline, a DVC file is created which you can push to your Git repository for versioning. We have included a few pipelines in `dvc_pipelines.sh` and we will illustrate the use case for DVC by running the finetuning pipeline below.


1. We create the finetune.dvc file by running the finetune pipeline. The finetune.dvc metafile will have to be commmited to the Git repository.

```
# finetune pipeline
dvc run -f dvc/finetune.dvc \
        -d src/finetune/config.py \
        -o test_finetune \
        python -m src.finetune.main
```

```
# on Git branch: "baseline_model"

git add dvc/finetune.dvc
git commit -m "Create finetune pipeline"
git push
```

In the future, we can always checkout the "baseline_model" Git branch to rerun the fine-tuning step. Git checkout ensures that we have the latest experiment code from our Git repository, while dvc checkout command will pull the latest copy of the data from our DVC remote. DVC remotes provide a central place to keep and share data and model files. With this remote storage, you can pull models and data files created by colleagues without spending time and resources to build or process them locally.

DVC does support several types of remote storage such as Amazon S3, and using remote storages are optional. DVC should work on your local computer out of the box. Feel free to check out the [official documentation](https://dvc.org/doc/command-reference/remote) to have a better understanding of DVC remotes.

```
git checkout baseline_model
dvc checkout
```

```
dvc repro finetune.dvc
```

2. Now imagine a scenario where we want to analyze the impact of the model results given a change in the fine-tuning parameters. We create a new Git branch and make the changes to the src/finetune/config.py file, rerun the pipeline and commit the "dvc/finetune.dvc" file to our new Git branch.

```
# run the finetune pipeline after making changes to src/finetune/config.py
dvc run -f dvc/finetune.dvc \
        -d src/finetune/config.py \
        -o test_finetune \
        python -m src.finetune.main
```

```
# on Git branch: "new_model"

git add dvc/finetune.dvc
git commit -m "Create new finetune pipeline with updated finetuning parameters"
git push
```

Subsequently, you can checkout the "new_model" branch to reproduce the same set of results and model weights with the updated finetuning parameters
```
git checkout new_model
dvc checkout
```

```
dvc repro finetune.dvc
```


<h1 align="center"> Acknowledgements </h1>

This project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG-RP-2019-050). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore.