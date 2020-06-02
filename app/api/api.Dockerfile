FROM tensorflow/tensorflow:latest-py3

COPY app/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD . /code
WORKDIR /code

CMD ["python", "app/api/main_es.py"]