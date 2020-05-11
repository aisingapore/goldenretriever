#!usr/bin/bash

# extract data from elasticsearch 
dvc run -f dvc/extract.dvc \
        -d src/dvc_pipeline_scripts/nrf_extract.py \
        -o data/nrf_test.csv \
        python src/dvc_pipeline_scripts/nrf_extract.py
        
# build index of response embeddings for faster lookup 
dvc run -f dvc/encode.dvc \
        -d src/dvc_pipeline_scripts/nrf_encode.py \
        -d data/nrf_test.csv \
        -o model_artefacts/nrf-data.pkl \
        -o model_artefacts/nrf.idx \
        python src/dvc_pipeline_scripts/nrf_encode.py --data data/nrf_test.csv --output_folder model_artefacts --index_prefix nrf --tfhub_module_url 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3'

# finetune pipeline


# make query pipeline
