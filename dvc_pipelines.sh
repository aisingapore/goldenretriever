#!usr/bin/bash

# extract data from elasticsearch 
dvc run -f dvc/extract.dvc \
        -d src/dvc_pipeline_scripts/pdpa_extract.py \
        -o data/nrf_test.csv \
        python src/dvc_pipeline_scripts/pdpa_extract.py
        
# build index of response embeddings for faster lookup 
dvc run -f dvc/encode.dvc \
        -d src/dvc_pipeline_scripts/pdpa_encode.py \
        -d data/pdpa2.csv \
        -o model_artefacts/pdpa-data.pkl \
        -o model_artefacts/pdpa.idx \
        python src/dvc_pipeline_scripts/pdpa_encode.py --data data/pdpa2.csv --output_folder model_artefacts --index_prefix pdpa --gr_model USEEncoder

# finetune pipeline
dvc run -f dvc/finetune.dvc \
        -d src/finetune/config.py \
        -o test_finetune \
        python -m src.finetune.main

# make query pipeline
