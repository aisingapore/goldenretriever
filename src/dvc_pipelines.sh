#/usr/bin/bash

dvc run -f extract.dvc \
        -d src/elasticsearch/extract.py \
        -o data/nrf_test.csv \
        python src/elasticsearch/extract.py
        
