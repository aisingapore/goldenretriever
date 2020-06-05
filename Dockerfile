FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y unixodbc-dev python3-dev

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD . /code
WORKDIR /code

EXPOSE 80

COPY wait-for-it.sh ./wait-for-it.sh
RUN chmod +x wait-for-it.sh