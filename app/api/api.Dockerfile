FROM tensorflow/tensorflow:latest-py3

COPY app/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# RUN apt-get -y install wget \
# && wget https://finetunedweights.blob.core.windows.net/finetuned02/weights.tar.gz \
# && tar -zxvf weights.tar.gz

ADD . /code
WORKDIR /code

CMD ["python", "app/api/main_es.py"]