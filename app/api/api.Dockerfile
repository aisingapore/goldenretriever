FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y unixodbc-dev python3-dev git

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD . /code
WORKDIR /code
RUN pip install -e .

CMD ["uvicorn", "app.api.main_es:app", "--host", "0.0.0.0", "--port", "80"]
