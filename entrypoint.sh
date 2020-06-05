#!/bin/bash
echo "Creating elasticsearch index"
python src/elasticsearch/create_doc_index.py elk:9200 data/pdpa2.csv pdpa-qa
python src/elasticsearch/create_querylog_index.py elk:9200 querylog
echo "Starting app"
uvicorn app.api.main_es:app --host "0.0.0.0" --port 80

