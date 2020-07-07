FROM ubuntu

# create the environment
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get -y install texlive-full latexmk

RUN apt-get update && apt-get -y install python3 python3-pip voikko-fi python3-libvoikko r-base git graphviz

# install pip packages


# install R packages
RUN R -e "install.packages(c('haven'), dependencies=TRUE, repos='http://cran.rstudio.com/')"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt

WORKDIR /speechrecognition
