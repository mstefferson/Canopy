FROM geodata/gdal
ADD requirements.txt /app/requirements.txt
# set working directory to /app/
WORKDIR /app/
# set up pip and install python packages
RUN apt-get update 
RUN pip install --upgrade pip
#RUN apt-get install python
#RUN apt-get install python3
#RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
# and vim
#RUN apt-get install -y vim
# create unprivileged user
RUN adduser --disabled-password --gecos ‘’ myuser
# Run initial command
CMD bash
