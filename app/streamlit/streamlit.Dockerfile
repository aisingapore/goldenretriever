FROM tensorflow/tensorflow:latest-py3

RUN apt-get update && apt-get install -y unixodbc-dev python3-dev git

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD . /code
WORKDIR /code
RUN pip install -e .

ENV PYTHONPATH "/code"
CMD ["streamlit", "run", "--server.port", "5000","--server.headless","true", "--browser.serverAddress","0.0.0.0", "--server.enableCORS", "false",  "app/streamlit/main.py"]