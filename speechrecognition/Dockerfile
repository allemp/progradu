FROM ubuntu

# create the environment
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get -y install texlive-full=2017.20180305-1 latexmk=1:4.41-1 

RUN apt-get update && apt-get -y install python3=3.6.7-1~18.04 python3-pip=9.0.1-2.3~ubuntu1.18.04.1 voikko-fi=2.2-1 python3-libvoikko=4.1.1-1.1 r-base=3.4.4-1ubuntu1 git graphviz

# install pip packages


# install R packages
RUN R -e "install.packages(c('haven'), dependencies=TRUE, repos='http://cran.rstudio.com/')"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

WORKDIR /speechrecognition
