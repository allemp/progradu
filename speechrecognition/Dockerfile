FROM ubuntu

# create the environment
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y install -y voikko-fi python-libvoikko

# install pip packages
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install nltk
#RUN python3 -m nltk.downloader finnish

WORKDIR /speechrecognition
