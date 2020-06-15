#!/bin/bash
echo "Creating elasticsearch index"
python src/elasticsearch/create_doc_index.py elk:9200 data/pdpa.csv pdpa-qa
python src/elasticsearch/create_querylog_index.py elk:9200 querylog
# echo "Creating simple nn index"
# python src/dvc_pipeline_scripts/pdpa_encode.py --data data/pdpa.csv --output_folder model_artefacts --index_prefix pdpa --gr_model USEEncoder
# echo "Starting app"
# uvicorn app.api.main_es:app --host "0.0.0.0" --port 80
# streamlit run app/streamlit/main.py

