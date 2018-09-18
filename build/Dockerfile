FROM python:3
ADD requirements.txt /app/requirements.txt
# set working directory to /app/
WORKDIR /app/
# set up pip 
RUN apt-get update 
RUN pip install --upgrade pip &&\
pip install --upgrade setuptools
# and vim
RUN apt-get install -y vim
# build gdal
RUN apt-get install build-essential python-all-dev &&\
wget http://download.osgeo.org/gdal/1.11.2/gdal-1.11.2.tar.gz &&\
tar xvfz gdal-1.11.2.tar.gz &&\
cd gdal-1.11.2 &&\
./configure --with-python &&\
make &&\
make install &&\
ldconfig &&\
# install python dependencies
RUN pip install -r requirements.txt
# create unprivileged user
RUN adduser --disabled-password --gecos ‘’ myuser
# Run initial command
CMD bash
