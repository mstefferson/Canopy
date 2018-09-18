FROM python:3
ADD requirements.txt /app/requirements.txt
# set working directory to /app/
WORKDIR /app/
# set up pip and vim
RUN apt-get update 
RUN pip install --upgrade pip
RUN apt-get install -y vim
# install python dependencies
RUN pip install -r requirements.txt
# create unprivileged user
RUN adduser --disabled-password --gecos ‘’ myuser
# Run initial command
CMD bash
