FROM tensorflow/tensorflow:latest-py3

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
&& curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list \
&& apt-get update \
&& ACCEPT_EULA=Y apt-get -y install msodbcsql17 unixodbc unixodbc-dev 

COPY app/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# RUN apt-get -y install wget \
# && wget https://finetunedweights.blob.core.windows.net/finetuned02/variables.tar.gz \
# && tar -zxvf variables.tar.gz

ADD . /code
WORKDIR /code

CMD ["python", "app/api/main.py", "-db", "db_cnxn_str.txt"]